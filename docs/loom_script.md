# Loom script: voice agent eval on tau2-bench retail (~5 min)

Read at a relaxed pace, roughly 140 words a minute. Screen cues are in *italics*; don't read them aloud.

---

**0:00 to 0:30: Which benchmark**

*Screen: the repo README.*

Hi, I'm Priyal. This is my take-home: a voice agent for the retail domain, evaluated on tau-bench. The first thing I checked was which tau-bench the assignment meant. The taubench.com link resolves to tau-cubed bench, which lives in Sierra's tau2-bench repo. The older tau-bench repo says in its own README that it's outdated. That matters, because only tau2-bench has the voice mode, the agent interface and the grader I built on.

**0:30 to 1:00: Architecture**

*Screen: the architecture diagram in the README.*

The assignment asks for my own choice of models, including speech-to-text and text-to-speech. tau2's realtime voice mode has no separate STT or TTS, and its cascaded LiveKit mode skips LiveKit's Agent entirely. So I built a real LiveKit Agent: Deepgram, GPT-4.1-mini, and OpenAI speech, wrapped as a turn-based tau2 agent. Every tool call goes back through tau2's own environment, so tau2 grades the conversation and the final database.

**1:00 to 1:30: Text-mode results**

*Screen: the report for `wide30`.*

In text mode, over thirty retail tasks, the agent passes sixty percent. I found three failure behaviors. First, missed actions: two thirds of tasks skip a required tool call, and the most common one is get product details. Second, in about half of the tasks the agent speaks and calls a tool in the same turn, which the prompt already forbids. Third, a counting bug, which I'll show you next.

**1:30 to 2:00: The counting bug**

*Screen: the task 2 transcript.*

In task two, the caller asks how many t-shirt options are available. The agent looks up the product and gets twelve variants back, but only ten of them are marked available. It tells the caller twelve. It counted everything instead of filtering out the unavailable ones first. Tasks three and four got the same question and failed in a related way: they said "several" and never gave a number at all.

**2:00 to 2:30: Audio bug, first attempt**

*Screen: the `baudio1` error.*

Then I turned on the audio path. Each user turn is spoken with text-to-speech and transcribed with Deepgram before the agent hears it. My first audio run died on the very first transcription with "attempted to use an HTTP session outside of a job context." Deepgram's plugin expects a shared HTTP session that LiveKit's job worker normally provides, and my agent runs outside a job worker. The language model worked fine because the OpenAI plugin manages its own client.

**2:30 to 3:00: Second and third attempts**

*Screen: `bridge.py` diff, then the test output.*

My second attempt opened a session around each speech call. That fixed turn one and broke turn two: Deepgram keeps the session after first use, and my wrapper had already closed it. The fix that held opens one session when the agent starts, runs every speech call inside it, and closes it once at the end. The regression test fails one way on the original, a different way on attempt two, and passes on the fix.

**3:00 to 3:30: Audio results**

*Screen: the `audio15` report.*

With audio working, the pass rate on fifteen tasks was thirty-three percent, with a word error rate of about thirty-nine percent. My first guess was that misheard names and zip codes caused the missed authentication steps. The transcripts proved me wrong: in four of five relevant tasks, transcription was near perfect. In the one real mishearing, "Mei" became "May", and the agent recovered. So audio mostly shows the same behavior bug as text.

**3:30 to 4:00: Prompt v2 and a fair test**

*Screen: `prompts/v2.md`.*

Next I wrote a second prompt that targets all three behaviors: a wrong-and-right example of talking during a tool call, taken from the agent's most common real violation, a rule to check product details before any quantity question, and a worked example of counting only available variants. Tasks two to four shaped this prompt, so testing on them would flatter it. I tested on seven held-out tasks instead: five, nine, twelve, seventeen, eighteen, twenty-six and twenty-seven.

**4:00 to 4:30: The result, a regression**

*Screen: `voice-tau compare` output.*

This is my strongest result, and it's a regression. On those seven tasks, the baseline scores zero point eight five seven and v2 scores zero point four two nine. Task twelve is the clean win: v2 removed exactly the two violations it targeted. Talking during tool calls dropped from five tasks to three. But tasks nine, eighteen and twenty-seven went from passing to failing. On tasks five and eighteen, v2 newly skipped get product details, the very call it made mandatory.

**4:30 to 5:00: What it means**

*Screen: `docs/DECISIONS.md`.*

So the fixes landed where I aimed them, and the extra instructions cost more elsewhere than they saved. If I had only tested on the tasks that inspired the prompt, I'd have reported a win. With another week, I'd apply the three fixes one at a time to see which one causes the new failures, and run more audio tasks. Every decision and bug along the way, with its evidence, is in the decision log. Thanks for watching.
