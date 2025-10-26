cd /hpctmp/e1351271/wkdbs/src
conda activate torch2
export PYTHONPATH=.

if [ -z "$OPENAI_API_KEY" ]; then
  echo "OPENAI_API_KEY is not set; export it before running this script."
  exit 1
fi

python -u utils/llm_judger.py
