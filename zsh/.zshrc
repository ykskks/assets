# ref: https://qiita.com/kinchiki/items/57e9391128d07819c321

# 環境変数
export your_env_var="xxx"
export PATH="/path/to/something:$PATH"

# autocorrectしない
unsetopt correct_all
unsetopt correct
DISABLE_CORRECTION="true"

# alias
alias ga="git add"
alias gb="git branch"
alias gbd="git branch -d"
alias gbdd="git branch -D"
alias gc="git commit -m"
alias gce="git commit --allow-empty -m"
alias gfo="git fetch origin"
alias gl="git log --oneline --graph"
alias gm="git merge"
alias gmv="git mv"
alias gp="git push"
alias gp="git push origin main"
alias gpl="git pull"
alias grao="git remote add origin"
alias grm="git rm"
alias gs="git status"
alias gsw="git switch"