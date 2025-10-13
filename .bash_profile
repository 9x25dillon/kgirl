#
# ~/.bash_profile
#

[[ -f ~/.bashrc ]] && . ~/.bashrc

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
