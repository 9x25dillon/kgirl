"""kgirl's mind: conversation memory, reflection, an agentic chat loop, spec -> code and long-form writing.

    conversations.py  every turn stored, searchable, summarized; commitments, topic tags, draft artifacts
    reflect.py        conversations -> summary, threads, commitments, facts, preferences, lessons, feedback credit
    mind.py           the chat agent: memory-aware context, tools, streaming
    budget.py         one token budget for the whole prompt: fixed blocks, priority cuts, what was cut
    builder.py        spec -> tested code by sampling and agreement; accepted results become exemplars
    writer.py         outline -> draft -> critique -> revise, resumable drafts in the user's style
    evolve.py         consolidation: curate habits, rewrite the user profile, export a fine-tune corpus
"""

from .builder import Builder, BuildResult
from .conversations import Commitment, Conversation, ConversationStore, Turn
from .evolve import EvolveReport, evolve, preview
from .mind import Mind, Reply
from .reflect import Reflection, Reflector
from .writer import Draft, Writer

__all__ = ["Builder", "BuildResult", "Commitment", "Conversation", "ConversationStore", "Turn", "EvolveReport",
           "evolve", "preview", "Mind", "Reply", "Reflection", "Reflector", "Draft", "Writer"]
