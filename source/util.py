import subprocess


def get_short_git_commit_hash():
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], stdout=subprocess.PIPE, text=True
    )
    return result.stdout.strip()
