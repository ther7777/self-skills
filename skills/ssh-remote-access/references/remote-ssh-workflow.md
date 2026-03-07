# Remote SSH Workflow

## Purpose

Use this reference when another AI needs to regain SSH access to the same environment and then edit code on the remote server.

## Environment Proven In This Session

- Local workspace: `d:\coding\DayValue`
- Local SSH config: `C:\Users\闈虫旦瀹嘰.ssh\config`
- Existing local keys:
  - `C:\Users\闈虫旦瀹嘰.ssh\id_rsa`
  - `C:\Users\闈虫旦瀹嘰.ssh\id_rsa.pub`
- `ssh-agent` was running locally.
- Tabby config path: `C:\Users\闈虫旦瀹嘰AppData\Roaming\tabby\config.yaml`

## Known Hosts

Configured aliases:

```sshconfig
Host host132
    HostName 10.112.161.132
    User hzl
    Port 1014
    PreferredAuthentications publickey,password
    IdentityFile C:/Users/闈虫旦瀹?.ssh/id_rsa

Host 3090
    HostName 10.112.17.69
    User dell
    Port 1014
    PreferredAuthentications publickey,password
    IdentityFile C:/Users/闈虫旦瀹?.ssh/id_rsa
```

Validated host:

- `3090` login succeeded with public key auth.
- Remote user: `dell`
- Remote home: `/home/dell`
- Likely repo path discovered: `/home/dell/FSL`

## What Failed

### Password login

Trying password auth against `3090` with `dell@10.112.17.69:1014` returned:

```text
Permission denied (publickey,password).
```

Interpret this as "host reachable, auth failed".

### Reading local SSH config without permission

Reading `C:\Users\闈虫旦瀹嘰.ssh\config` required elevated local access in this environment.

### Broken local SSH config entry

The original `host132` entry had:

- A commented-out `IdentityFile`
- A path containing mojibake / encoding damage

That config was unreliable for automated use and had to be rewritten.

### Tabby config did not contain saved SSH profiles

`config.yaml` contained known host fingerprints, but no saved `profiles`. Do not assume Tabby stores reusable auth details there.

## Working Procedure

1. Read local SSH config and identify the target alias.
2. Inspect `~/.ssh` and confirm whether a private key already exists.
3. Confirm the private key is usable:

```powershell
ssh-keygen -y -f C:\Users\闈虫旦瀹嘰.ssh\id_rsa
```

4. Confirm `ssh-agent` is running and list loaded identities:

```powershell
Get-Service ssh-agent
ssh-add -l
```

5. If the user already has a server session open in Tabby, use that session to append the local public key:

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
grep -qxF '<local-public-key>' ~/.ssh/authorized_keys 2>/dev/null || echo '<local-public-key>' >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

6. Back up and repair the local SSH config so the host explicitly uses the working key.
7. Validate with:

```powershell
ssh -o BatchMode=yes -o PreferredAuthentications=publickey -o PasswordAuthentication=no 3090 "echo __CODEX_CONNECTED__ && whoami && pwd"
```

Expected output pattern:

```text
__CODEX_CONNECTED__
dell
/home/dell
```

8. After SSH succeeds, inspect remote repositories:

```bash
find ~ -maxdepth 2 \( -name .git -o -name README.md -o -name pyproject.toml -o -name package.json \) 2>/dev/null | sed 's#/\.git##' | sort -u
```

## Pitfalls

- Do not create a new key until checking whether an existing key already exists.
- Do not rely on password auth for repeatable remote edits; it breaks non-interactive automation.
- Do not copy the private key to the server. Only append the public key.
- Do not assume Tabby's success means `ssh.exe` is ready; validate with `ssh.exe` explicitly.
- Do not leave `IdentityFile` commented out in `~/.ssh/config`.
- If the target machine changes, the same procedure still applies, but the public key installed on the server must match the private key present on that machine.

## Reuse On Another Computer

If another computer needs access:

1. Ensure that computer has its own private/public key pair, or securely copy the intended existing private key and public key to that machine.
2. Load the private key into that machine's `ssh-agent`.
3. Add that machine's public key to the server's `authorized_keys`.
4. Update that machine's `~/.ssh/config` with the correct `IdentityFile`, host alias, username, hostname, and port.
5. Re-run the non-interactive SSH validation command from that machine.

Do not assume a skill alone grants access. The remote server must trust the public key from the current machine.
