"""AzCopy adapter; deliberately independent of SurgeryOCR and GPU packages."""
import argparse
import getpass
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import parse_qs, quote, urlsplit, urlunsplit

DEFAULT_CONFIG = Path.home() / '.config' / 'zettabyte-transfer' / 'config.json'


def validate_url(value):
    if not isinstance(value, str):
        raise ValueError("SAS URL 必須是字串。")
    parts = urlsplit(value.strip())
    if (parts.scheme != 'https' or not parts.hostname
            or not parts.hostname.endswith('.blob.core.windows.net')
            or parts.username or parts.password or parts.fragment
            or len(parts.path.strip('/').split('/')) != 1 or not parts.path.strip('/')):
        raise ValueError('SAS URL 必須是 HTTPS Azure Blob 容器網址（不能含子目錄）。')
    token = parse_qs(parts.query)
    if not token.get('sig') or token.get('sr') != ['c']:
        raise ValueError('需要原信完整的 container SAS URL（含 sig 與 sr=c）。')
    return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip('/'), '', '')), parts.query


def remote_path(value):
    if (not value or value.startswith('/') or '\\' in value
            or any(part in ('', '.', '..') for part in value.split('/'))
            or any(c in value for c in '*?#[\x00\r\n')):
        raise ValueError('遠端路徑須為相對路徑，不可含 ..、萬用字元或空白路徑段。')
    return str(PurePosixPath(value))


def load_url(config):
    value = os.environ.get('ZETTABYTE_SAS_URL')
    if not value:
        try:
            value = json.loads(config.read_text(encoding='utf-8'))['sas_url']
        except (OSError, ValueError, KeyError, TypeError):
            raise ValueError('尚未設定憑證：執行 configure 或設定 ZETTABYTE_SAS_URL。') from None
    return validate_url(value)


def redact(value):
    value = re.sub(r'https://[^\s\"\']+\?[^\s\"\']+', '[SAS URL hidden]', value)
    return re.sub(r'(?i)(sig=)[^&\s]+', r'\1[hidden]', value)


def execute(command, dry_run=False):
    print(redact(' '.join(command)), flush=True)
    if dry_run:
        return 0
    if not shutil.which(command[0]):
        raise ValueError('找不到 AzCopy；請安裝 AzCopy v10 或使用 --azcopy 指定執行檔。')
    # Never stream raw AzCopy diagnostics: they may contain the SAS URL.
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, errors='replace')
    try:
        for line in process.stdout:
            print(redact(line), end='', flush=True)
        return process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        return 130


def parser():
    p = argparse.ArgumentParser(description='Zettabyte 資料／模型傳輸工具（AzCopy v10）')
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG, help='憑證設定檔位置')
    p.add_argument('--azcopy', default=os.environ.get('ZETTABYTE_AZCOPY', 'azcopy'))
    p.add_argument('--dry-run', action='store_true', help='只顯示遮蔽憑證後的指令')
    sub = p.add_subparsers(dest='action', required=True)
    sub.add_parser('configure', help='安全輸入完整 SAS URL 並存至設定檔')
    sub.add_parser('doctor', help='檢查設定與 AzCopy（不連線）')
    ls = sub.add_parser('list', help='列出 Azure 中的檔案（不代表 PVC 同步狀態）')
    ls.add_argument('remote', nargs='?', default='')
    for name, help_text in [('upload', '本機／IDE → 容器根目錄，供自動同步至 PVC'),
                            ('publish', 'IDE → outbound/，供本機下載')]:
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument('source', type=Path)
        cmd.add_argument('remote', help='遠端相對路徑，例如 jobs/job001/video.mp4 或 models/v1')
        cmd.add_argument('--overwrite', action='store_true', help='允許覆寫同名檔案')
    dl = sub.add_parser('download', help='從 outbound/ 下載到本機；--container 可讀其他位置')
    dl.add_argument('remote', help='outbound 下相對路徑（不要加 outbound/）')
    dl.add_argument('destination', type=Path)
    dl.add_argument('--container', action='store_true', help='改從容器根目錄讀取')
    dl.add_argument('--overwrite', action='store_true')
    for name in ('status', 'resume'):
        cmd = sub.add_parser(name, help='查看／續傳本機 AzCopy job（需保留 AzCopy job plan）')
        cmd.add_argument('job_id')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.action == 'configure':
            if args.dry_run:
                raise ValueError('configure 不支援 --dry-run；不會寫入設定。')
            value = getpass.getpass('完整 SAS URL（不回顯）: ').strip()
            validate_url(value)
            args.config.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(dir=args.config.parent, prefix='.sas-')
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump({'sas_url': value}, f)
                os.replace(temporary, args.config)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            print('憑證已儲存；請勿提交設定檔至 Git。')
            return 0
        base, token = load_url(args.config)
        if args.action == 'doctor':
            print('SAS URL 結構：有效（尚未驗證伺服器授權）')
            print('AzCopy：' + ('已找到' if shutil.which(args.azcopy) else '未安裝'))
            return 0 if shutil.which(args.azcopy) else 1
        def url(path=''):
            return base + ('/' + quote(path, safe='/') if path else '') + '?' + token
        if args.action in ('status', 'resume'):
            if not re.fullmatch(r'[0-9a-fA-F-]{36}', args.job_id):
                raise ValueError('job_id 必須是 AzCopy 的 UUID。')
            command = [args.azcopy, 'jobs', 'show' if args.action == 'status' else 'resume', args.job_id]
            if args.action == 'resume':
                command += ['--source-sas=' + token, '--destination-sas=' + token]
        elif args.action == 'list':
            command = [args.azcopy, 'list', url(remote_path(args.remote) if args.remote else '')]
        else:
            path = remote_path(args.remote)
            if args.action == 'upload':
                if path.split('/')[0].lower() == 'outbound':
                    raise ValueError('upload 不可使用 outbound/；請改用 publish。')
            elif args.action == 'publish' or not args.container:
                if path.split('/')[0].lower() == 'outbound':
                    raise ValueError('請提供 outbound/ 下的相對路徑，不要重複前綴。')
                path = 'outbound/' + path
            if args.action in ('upload', 'publish'):
                if not args.source.exists() or args.source.is_symlink():
                    raise ValueError('來源不存在或為符號連結。')
                source, destination = str(args.source.resolve()), url(path)
                directory = args.source.is_dir()
            else:
                source, destination = url(path), str(args.destination.resolve())
                directory = True
            command = [args.azcopy, 'copy', source, destination,
                       '--overwrite=' + ('true' if args.overwrite else 'false')]
            if directory:
                command += ['--recursive=true', '--as-subdir=false']
            if args.action in ('upload', 'publish'):
                command += ['--put-md5']
            else:
                command += ['--check-md5=FailIfDifferent']
        return execute(command, args.dry_run)
    except (ValueError, OSError) as exc:
        print('錯誤：' + redact(str(exc)), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
