#!/usr/bin/env bash
set -euo pipefail

HOME_DIR="${HOME:-/root}"

write_ssh_key() {
  local ssh_dir="${HOME_DIR}/.ssh"
  local key_path="${SSH_KEY_PATH:-${ssh_dir}/id_ed25519}"
  local host="${GIT_SSH_HOST:-github.com}"

  if [[ -n "${SSH_PRIVATE_KEY_B64:-}" || -n "${SSH_PRIVATE_KEY:-}" ]]; then
    mkdir -p "${ssh_dir}"
    chmod 700 "${ssh_dir}"

    if [[ -n "${SSH_PRIVATE_KEY_B64:-}" ]]; then
      printf '%s' "${SSH_PRIVATE_KEY_B64}" | tr -d '\r' | base64 -d > "${key_path}"
    else
      printf '%s\n' "${SSH_PRIVATE_KEY}" | tr -d '\r' > "${key_path}"
    fi
    chmod 600 "${key_path}"

    if [[ -n "${SSH_KNOWN_HOSTS:-}" ]]; then
      printf '%s\n' "${SSH_KNOWN_HOSTS}" >> "${ssh_dir}/known_hosts"
    else
      ssh-keyscan -H "${host}" >> "${ssh_dir}/known_hosts" 2>/dev/null || true
    fi
    chmod 600 "${ssh_dir}/known_hosts" || true
  fi
}

write_aws_creds() {
  if [[ -n "${AWS_ACCESS_KEY_ID:-}" && -n "${AWS_SECRET_ACCESS_KEY:-}" ]]; then
    local aws_dir="${HOME_DIR}/.aws"
    local cred_file="${AWS_SHARED_CREDENTIALS_FILE:-${aws_dir}/credentials}"
    local config_file="${AWS_CONFIG_FILE:-${aws_dir}/config}"
    local region="${AWS_DEFAULT_REGION:-${AWS_REGION:-}}"

    mkdir -p "${aws_dir}"
    chmod 700 "${aws_dir}"

    {
      echo "[default]"
      echo "aws_access_key_id=${AWS_ACCESS_KEY_ID}"
      echo "aws_secret_access_key=${AWS_SECRET_ACCESS_KEY}"
      if [[ -n "${AWS_SESSION_TOKEN:-}" ]]; then
        echo "aws_session_token=${AWS_SESSION_TOKEN}"
      fi
    } > "${cred_file}"
    chmod 600 "${cred_file}"

    if [[ -n "${region}" ]]; then
      {
        echo "[default]"
        echo "region=${region}"
      } > "${config_file}"
      chmod 600 "${config_file}"
    fi
  fi
}

write_ssh_key
write_aws_creds

exec "$@"
