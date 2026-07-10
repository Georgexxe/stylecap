# Style: HUMOROUS-TECH

## Voice
Developer group-chat energy. Maps whatever happens in the clip onto software/AI/startup
tropes — but the mapping must still accurately describe the clip (accuracy gate applies to jokes).

## Trope bank (map clip events onto these)
race condition · merge conflict · "works on my machine" · deploying to prod on Friday ·
rollback · infinite loop · LGTM · standup update · edge case · technical debt · hotfix ·
99% test coverage, failed in prod · O(n²) · garbage collection · rate-limited ·
hallucination/LLM jokes · "have you tried turning it off and on again"

## Rules
- One central tech metaphor per caption, sustained — not five mixed metaphors.
- The clip's actual events must be recoverable from the caption (specificity ≥2 details).
- Jargon is the humor vehicle, but a non-dev should still follow the story.
- 1–3 sentences.

## Banned
Metaphor soup (>2 tropes) · jokes that abandon the clip's content · forced acronyms ·
explaining the reference.

## Gold exemplars (facts → caption)

**Facts:** projector fails at 0:41; presenter switches to flip chart; applause.
**Caption:** Production incident at 0:41: projector returns a fatal error mid-demo. Presenter
executes a flawless rollback to the legacy flip-chart stack — zero downtime, stakeholders
approve the hotfix with applause.

**Facts:** cat knocks third glass off counter while looking at camera.
**Caption:** After two successful test runs, Whiskers ships glass-deletion v3 straight to prod,
maintaining eye contact with the incident reviewer the entire time. No rollback planned.

**Facts:** jogger reties shoe; pigeon steals energy bar; chase.
**Caption:** Classic race condition: while the jogger's main thread blocks on shoelace I/O,
a pigeon process acquires the energy bar and refuses to release the lock. The chase that
follows is best described as unoptimized.

**Facts:** kitchen; cook flips pancake; pancake lands on floor; dog takes it.
**Caption:** The pancake deployment misses the plate environment and lands directly in floor production. The dog consumes the failed build before rollback can begin.

**Facts:** football pitch; goalkeeper saves shot; celebrates; ball rolls backward into goal.
**Caption:** The keeper marks the save ticket resolved before the ball process has actually terminated. One background rollback later, the goal ships anyway.

**Facts:** city pavement in heavy rain; umbrella turns inside out; pedestrian keeps walking.
**Caption:** Wind load exceeds the umbrella's documented API limits, forcing an unscheduled inversion. The pedestrian selects “continue anyway” and walks on through the rain.

**Facts:** factory table; robot arm sorts blue and red blocks; red block jams conveyor; arm stops.
**Caption:** The robot handles blue and red inputs until one red block triggers a blocking I/O call on the conveyor. No exception is caught; the entire arm simply waits.

**Facts:** mountain trail; hiker reaches waterfall overlook; rainbow visible in spray.
**Caption:** The hiker completes the mountain traversal and unlocks the waterfall view. The spray renderer then enables rainbow mode at maximum settings.
