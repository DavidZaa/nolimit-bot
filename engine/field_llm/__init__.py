"""Opponents written the way other teams will write theirs.

Every other bot in engine/field.py is our own guess at what the field looks
like. These four were written by fresh AI coding sessions that knew nothing
about our bot: each got one prompt a team might type, plus the engine
interface (bot/state.py) and the tournament constraints, and nothing else.
They are kept exactly as written, so they play with their authors' real
strengths and blind spots. Each module's class is `Bot`.

  llm1  "Write a Python poker bot for 8-player no-limit hold'em sit-and-go
         tournaments. Implement act(state) -> Action. It has to decide in under
         1 ms. Make it as strong as possible."
  llm2  "Write a tight-aggressive poker bot in Python for an 8-player NLHE
         sit-and-go with rising blinds."
  llm3  "Write a poker bot that uses Monte Carlo equity to decide whether to
         call, raise or fold."  It simulates until a wall-clock deadline, so its
         play varies slightly with machine speed.
  llm4  "Write a push/fold poker bot for short-stacked sit-and-go tournaments
         using Nash ranges and ICM."  Told the points are 50/30/20.
"""
