"""kgirl's mind: conversation memory, reflection, an agentic chat loop, spec -> code and long-form writing.

    conversations.py  every turn stored, searchable, summarized
    reflect.py        conversations -> summary, facts, preferences, lessons, feedback credit
    mind.py           the chat agent: memory-aware context, tools, streaming
    builder.py        spec -> tested code by sampling and agreement; accepted results become exemplars
    writer.py         plan -> draft -> critique -> revise
    evolve.py         consolidation: curate habits, rewrite the user profile, export a fine-tune corpus
"""

from .builder import Builder, BuildResult, edit_build
from .conversations import Conversation, ConversationStore, Turn
from .evolve import EvolveReport, evolve
from .mind import Mind, Reply
from .reflect import Reflection, Reflector
from .writer import Draft, Writer

__all__ = ["Builder", "BuildResult", "edit_build", "Conversation", "ConversationStore", "Turn", "EvolveReport", "evolve",
           "Mind", "Reply", "Reflection", "Reflector", "Draft", "Writer"]
