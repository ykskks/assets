# ref: https://atsashimipy.hatenablog.com/entry/2019/10/14/054338

# 事前に設定したい仮想環境の情報を以下のファイルにエクスポート
conda env export > environment.yaml

# ~/.bash_profileに以下を追加

# Auto activate conda environments
function conda_auto_env() {
  if [ -e "environment.yaml" ]; then
    ENV_NAME=$(head -n 1 environment.yaml | cut -f2 -d ' ')
    # Check if you are already in the environment
    if [[ $CONDA_PREFIX != *$ENV_NAME* ]]; then
      # Try to activate environment
      source activate $ENV_NAME &>/dev/null
    fi
  fi
}

export PROMPT_COMMAND="conda_auto_env;$PROMPT_COMMAND"