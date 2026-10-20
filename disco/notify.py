#!/usr/bin/env python3
"""Fan-out notifier for the benchmark watchdog.

Reads channel credentials from notify.env (NOT sourced as shell -- parsed here so
a stray quote / inline comment / duplicate key cannot break delivery) and sends a
subject+body to EVERY configured channel: ntfy.sh, Slack webhook, Discord webhook,
and SMTP email. Prints one line per channel (ok/fail) and exits 0 if at least one
channel delivered, 1 if all failed. Secrets are never printed.

Usage:
  notify.py "<subject>" "<body>"            # read body from arg
  echo "<body>" | notify.py "<subject>"      # read body from stdin
  notify.py --env /path/notify.env "<subject>" "<body>"
"""
import os, sys, json, ssl, smtplib, urllib.request
from email.mime.text import MIMEText

DEFAULT_ENV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notify.env")


def parse_env(path):
    """Parse KEY=VALUE lines robustly. Collects duplicate keys into a list.
    Strips surrounding quotes and trailing ' #inline comment' noise."""
    multi = {}
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            # strip a trailing inline comment that follows a closing quote or space
            # e.g.  topic" #change me   -> topic   ;  val # note -> val
            if '"' in v:
                # take content up to the first stray closing quote if unbalanced
                if v.count('"') % 2 == 1:
                    v = v.split('"')[0] if v[0] != '"' else v[1:].split('"')[0]
            # drop ' #...' inline comment (space then hash) if not inside a URL
            for sep in (' #', '\t#'):
                if sep in v:
                    v = v.split(sep)[0]
            v = v.strip().strip('"').strip("'").strip()
            if not v:
                continue
            multi.setdefault(k, []).append(v)
    return multi


# Discord sits behind Cloudflare, which 403s (error 1010) the default python
# urllib User-Agent. A browser-like UA passes.
_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"


def _post_json(url, payload, timeout=20):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status


def send_slack(url, subject, body):
    return _post_json(url, {"text": f"*{subject}*\n{body}"})


def send_discord(url, subject, body):
    # Discord: 2000 char limit on content
    content = f"**{subject}**\n{body}"
    if len(content) > 1900:
        content = content[:1900] + "\n…(truncated)"
    return _post_json(url, {"content": content})


def send_ntfy(base, topic, subject, body, priority):
    url = base.rstrip("/") + "/" + topic
    req = urllib.request.Request(url, data=body.encode(),
                                 headers={"Title": subject,
                                          "Priority": priority or "default"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.status


def send_telegram(token, chat_id, subject, body):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    text = f"*{subject}*\n{body}"
    if len(text) > 4000:
        text = text[:4000] + "\n…(truncated)"
    # Markdown can choke on stray underscores/brackets in dataset names/logs; send
    # as plain text to avoid 400s (reliability over formatting for an alert bot).
    return _post_json(url, {"chat_id": chat_id, "text": text})


def send_email(env, subject, body):
    host = env["SMTP_HOST"][0]
    port = int(env.get("SMTP_PORT", ["587"])[0])
    user = env["SMTP_USER"][0]
    pw = env["SMTP_PASS"][0]
    frm = env.get("SMTP_FROM", [user])[0]
    to = env.get("NOTIFY_EMAIL", [user])[0]
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = frm
    msg["To"] = to
    ctx = ssl.create_default_context()
    with smtplib.SMTP(host, port, timeout=30) as s:
        s.starttls(context=ctx)
        s.login(user, pw)
        s.sendmail(frm, [to], msg.as_string())
    return 200


def main():
    argv = sys.argv[1:]
    env_path = DEFAULT_ENV
    if argv and argv[0] == "--env":
        env_path = argv[1]
        argv = argv[2:]
    subject = argv[0] if argv else "notification"
    body = argv[1] if len(argv) > 1 else (sys.stdin.read() if not sys.stdin.isatty() else "")
    env = parse_env(env_path)

    results = []
    for url in env.get("NOTIFY_WEBHOOK_URL", []):
        kind = "slack" if "hooks.slack.com" in url else ("discord" if "discord" in url else "webhook")
        try:
            st = send_slack(url, subject, body) if kind != "discord" else send_discord(url, subject, body)
            results.append((kind, True, st))
        except Exception as e:
            results.append((kind, False, f"{type(e).__name__}: {str(e)[:80]}"))

    if env.get("NOTIFY_NTFY_TOPIC") and env.get("NOTIFY_NTFY_BASE"):
        try:
            st = send_ntfy(env["NOTIFY_NTFY_BASE"][0], env["NOTIFY_NTFY_TOPIC"][0],
                           subject, body, env.get("NOTIFY_NTFY_PRIORITY", ["default"])[0])
            results.append(("ntfy", True, st))
        except Exception as e:
            results.append(("ntfy", False, f"{type(e).__name__}: {str(e)[:80]}"))

    tg_token = env.get("NOTIFY_TELEGRAM_TOKEN", [None])[0]
    tg_chat = env.get("NOTIFY_TELEGRAM_CHAT_ID", [None])[0]
    if tg_token and tg_chat:
        try:
            st = send_telegram(tg_token, tg_chat, subject, body)
            results.append(("telegram", True, st))
        except Exception as e:
            results.append(("telegram", False, f"{type(e).__name__}: {str(e)[:80]}"))
    elif tg_token and not tg_chat:
        results.append(("telegram", False, "no chat_id yet (message the bot once)"))

    if env.get("SMTP_HOST") and env.get("SMTP_PASS"):
        try:
            st = send_email(env, subject, body)
            results.append(("email", True, st))
        except Exception as e:
            results.append(("email", False, f"{type(e).__name__}: {str(e)[:80]}"))

    any_ok = False
    for kind, ok, info in results:
        print(f"  {kind:8s} {'OK' if ok else 'FAIL'} ({info})")
        any_ok = any_ok or ok
    if not results:
        print("  no channels configured")
    sys.exit(0 if any_ok else 1)


if __name__ == "__main__":
    main()
