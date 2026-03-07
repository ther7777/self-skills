---
name: ssh-remote-access
description: Prepare and verify SSH access to remote Linux servers before editing remote code or running experiments. Use when Codex needs to read local SSH config, inspect existing keys, reuse Tabby or local SSH settings, install a public key from an already-open server terminal, debug password/key authentication failures, or confirm non-interactive SSH access to a named host such as `3090` or `host132`.
---

# SSH Remote Access

Use this skill to make remote SSH access reliable before changing remote code. Favor key-based login over password login so later `ssh`, `git`, file copy, and experiment commands work non-interactively.

## Workflow

1. Read the local SSH config and identify the target host alias, username, hostname, and port.
2. Inspect `~/.ssh` for existing keys before generating new ones. Prefer reusing an existing private key if one already works locally.
3. Check whether `ssh-agent` is running and whether the intended key is already loaded.
4. If the user already has a working terminal in Tabby or another SSH client, use that server terminal to append the local public key into `~/.ssh/authorized_keys`.
5. Update the local SSH config so the host explicitly uses the intended `IdentityFile` and allows `publickey,password` in that order.
6. Validate with a non-interactive SSH command using `BatchMode=yes` and public-key auth only.
7. Only after SSH succeeds, inspect the remote repository and start code changes.

## Rules

- Never expose or copy the private key. Read only the public key for installation on the server.
- Prefer `ssh.exe` for validation even if the user normally logs in via Tabby. Tabby can be a useful source of hostnames and proof that the server is reachable, but `ssh.exe` is the path Codex can automate.
- If password login fails but the host is reachable, treat it as an auth problem, not a network problem.
- If a host alias exists in `~/.ssh/config`, validate against that alias rather than rebuilding the full command each time.
- Keep a backup before editing the local SSH config.

## Quick Checks

Run these checks locally:

```powershell
Get-Content $HOME\.ssh\config
Get-ChildItem $HOME\.ssh -Force
Get-Service ssh-agent
ssh-add -l
ssh-keygen -y -f $HOME\.ssh\id_rsa
```

Validate the target host non-interactively:

```powershell
ssh -o BatchMode=yes -o PreferredAuthentications=publickey -o PasswordAuthentication=no <host-alias> "echo __SSH_OK__ && whoami && pwd"
```

## Install The Public Key From An Existing Server Session

If the user already has a working terminal on the server, read the local public key and append it remotely:

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
grep -qxF '<local-public-key>' ~/.ssh/authorized_keys 2>/dev/null || echo '<local-public-key>' >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

Replace `<local-public-key>` with the full contents of the local `.pub` file.

## References

- Read [remote-ssh-workflow.md](references/remote-ssh-workflow.md) for the tested workflow, concrete commands, and pitfalls from this project.
