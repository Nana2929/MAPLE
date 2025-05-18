# Example of Google Drive
micromamba activate uie
gdown 12Dkh6KLDPvXrkQ1I-1xLqODQSYjkwnvs && unzip uie-base-en.zip
gdown 15OFkWw8kJA1k2g_zehZ0pxcjTABY2iF1 && unzip uie-large-en.zip
mkdir uie_checkpoints
mv uie-base-en uie-large-en uie_checkpoints
rm -rf uie-base-en.zip uie-large-en.zip