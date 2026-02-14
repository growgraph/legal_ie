#!/usr/bin/env bash
shopt -s globstar nullglob

echo "filename,position,mention" > mentions.csv

for pdf in **/*.pdf; do
  filename=$(basename "$pdf")

  # Extract all text
  text=$(pdftotext "$pdf" -)

  # More precise pattern that captures the complete location name
  echo "$text" | grep -bPoi "(?:cour|tribunal supérieur)\s+d['']appel\s+(?:d['']|de(?:\s+la)?|du|des)\s+(?:[A-ZÉÈÎÏÇÀÂÄÊËÎÏÔÖÙÛÜ][a-zéèêîïçàâäêëîïôöùûü]*(?:[-''][A-ZÉÈÎÏÇÀÂÄÊËÎÏÔÖÙÛÜ]?[a-zéèêîïçàâäêëîïôöùûü]+)*(?:\s+(?:de|d['']|la|le|les|du|des|sur)\s+[A-ZÉÈÎÏÇÀÂÄÊËÎÏÔÖÙÛÜ][a-zéèêîïçàâäêëîïôöùûü]*(?:[-''][A-ZÉÈÎÏÇÀÂÄÊËÎÏÔÖÙÛÜ]?[a-zéèêîïçàâäêëîïôöùûü]+)*)*)(?=\W|$)" |
  while IFS=: read -r pos mention; do
    # Additional filtering to remove problematic matches
    if ! echo "$mention" | grep -Pqi "(?:renvoi|et sa|cassation)\s*$" && 
       [[ $mention =~ [A-ZÉÈÎÏÇÀÂÄÊËÎÏÔÖÙÛÜ][a-zéèêîïçàâäêëîïôöùûü]+$ ]]; then
      echo "\"$filename\",$pos,\"$mention\"" >> mentions.csv
    fi
  done
done