# Experiment Overview

## multiturn_jailbreak_steering

Establishes whether a guard placed in front of an LLM stops multi-turn jailbreaks without disrupting normal use. Every guard is replayed, turn by turn, on recorded multi-turn attack conversations whose turns carry a judge's verdict on the LLM's response, and on benign multi-turn sessions. The benefit side is how many attack conversations still contain a jailbroken turn that got through (averaged over three attack strategies, plus the worst strategy among four). The constraint side is what the guard costs: benign knowledge and chat turns it filters, benign but sensitive-sounding prompts it refuses, and attack-conversation turns it filters although the LLM answered them safely. Detecting jailbroken turns at the turn level is reported as an F1. Guards that block almost nothing, refuse most benign traffic, or filter most of an attack conversation regardless of content fail the run.

## prompt_harmfulness_detection

Establishes whether the same guard, used as a prompt classifier, separates harmful requests from benign ones that superficially resemble them (XSTest safe/unsafe contrast prompts, JailbreakBench benign/harmful behaviours), each asked after some benign conversation. Scored as F1 of the filter decision. It complements the first experiment: a guard can lower attack success by filtering broadly, and this measures whether its decisions track harmfulness itself.
