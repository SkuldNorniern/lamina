#!/bin/bash
# Interactive executable cleaner with directory blacklist — refined version

BLACKLIST=(
  "./benchmarks"
  "./marks"
  "./.git"
  "./node_modules"
  "./target"
)

echo "Scanning for compiled executables (skipping blacklisted directories)..."
echo

find . -type f -perm +111 | while read -r f; do
  skip=false
  for dir in "${BLACKLIST[@]}"; do
    case "$f" in
      "$dir"/*) skip=true; break;;
    esac
  done

  if [ "$skip" = false ]; then
    # Detect true binaries (Mach-O, ELF) but skip scripts like Python/Shell
    desc=$(file "$f")
    if echo "$desc" | grep -Eq "Mach-O|ELF"; then
      exec_flag=true
    elif echo "$desc" | grep -Eq "script text executable"; then
      # exclude common interpreted languages
      if echo "$desc" | grep -Eq "Python|Bourne|POSIX|bash|zsh|Perl|Ruby"; then
        exec_flag=false
      else
        exec_flag=true
      fi
    else
      exec_flag=false
    fi

    if [ "$exec_flag" = true ]; then
      echo "[candidate] $f ($desc)"
      read -p "Remove this file? [y/N]: " choice < /dev/tty
      case "$choice" in
        [Yy]* ) rm -v "$f";;
        * ) echo "Skipped $f";;
      esac
    fi
  fi
done

