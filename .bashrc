#
# ~/.bashrc
#

# If not running interactively, don't do anything
[[ $- != *i* ]] && return

alias ls='ls --color=auto'
alias grep='grep --color=auto'
PS1='[\u@\h \W]\$ '

# >>> juliaup initialize >>>

# !! Contents within this block are managed by juliaup !!

case ":$PATH:" in
    *:/home/kill/.juliaup/bin:*)
        ;;

    *)
        export PATH=/home/kill/.juliaup/bin${PATH:+:${PATH}}
        ;;
esac

# <<< juliaup initialize <<<
vm.swappiness = 10
vm.vfs_cache_pressure = 50
export PATH=~/.npm-global/bin:$PATH
