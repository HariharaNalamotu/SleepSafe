# SleepSafe 


**Wearable audio analysis for better sleep awareness at home.** SleepSafe connects a Bluetooth headset to a Raspberry Pi to record nighttime audio, identify breathing and sound patterns, and present the results through a web dashboard. > **Main code:** The current implementation is in the [`hari` branch](https://github.com/HariharaNalamotu/SleepSafe/tree/hari). Switch to that branch to access the main project code. ## Overview SleepSafe helps people understand nighttime breathing and sounds through timestamped events and readable session reports. Our initial focus is sleep wellness and personal awareness, with a longer-term ambition to support clinician-directed home assessments. 

Inspiration
The most important sounds of your night happen while you are asleep. If you are worried about your breathing, it is difficult to turn that concern into something concrete to discuss with a clinician. Access, cost, and the inconvenience of formal sleep testing can make that first step feel out of reach.

SleepSafe began with a simple idea: make it easier to understand breathing during sleep from home. This build develops that idea into a connected audio-analysis prototype, with a wellness-first product direction focused on understanding sleep habits and nighttime sounds. We want to help people notice patterns and ask better questions about their sleep, without pretending that a headset can replace a clinical sleep study.

What it does
Put on a Bluetooth headset, connect it to our Raspberry Pi, and press Start recording on the web dashboard. The Pi captures audio and uploads it in 10-second chunks. Press Stop recording, and a Databricks workflow turns those chunks into timestamped breathing and sound events, a session summary, and a plain-language report.

The dashboard shows estimated breathing events per hour, estimated sleep time, snoring, an interactive event timeline, and expandable event details. A Databricks-hosted companion agent exposes the session data with classification and question-answering capabilities. The public dashboard currently provides recording controls and reports; the companion agent is a separate app.

Our near-term product direction is sleep wellness and personal awareness, not medical diagnosis or treatment. The current research prototype includes experimental breathing-event estimates; these are not clinically validated diagnostic results. Results from short recordings are preliminary, and consumer-headset performance still needs validation. A wellness release will require aligning its features and claims with that intended use.

How we built it
Edge capture: A Raspberry Pi acts as an edge computing device at the bedside. A Python agent uses PipeWire and FFmpeg to capture the Bluetooth headset microphone and process it on-device: resampling to 16 kHz mono and band-pass filtering (60 Hz–7 kHz) to cut rumble and hiss while keeping quiet breathing sounds. Audio is spooled locally in 10-second chunks and uploaded to a Databricks Unity Catalog Volume as recording continues, so a network drop doesn't lose data. The Pi uses outbound HTTPS to receive dashboard commands and publish its status, so it needs no inbound internet connection.

Learned audio model: We used technician-scored PSG-Audio recordings from 79 subjects, split by subject into 59 training, 10 validation, and 10 test subjects. A frozen PANNs CNN14 encoder produces per-second acoustic features; a compact bidirectional GRU head learns apnea, hypopnea, snoring, and sleep/wake patterns. Post-processing converts predictions into timestamped events. AudioSet-based gates reduce speech being mistaken for sleep or snoring.

Databricks workflow: Stopping a session hands off from the edge to the cloud: a serverless CPU job assembles the timestamped chunks, normalizes loudness, runs inference, and writes structured session and event data to Delta tables. A Databricks-hosted language model, databricks-gpt-oss-120b, turns that output into a non-diagnostic report. The companion Databricks App provides classification and Q&A over the session data. Databricks is the storage, processing, and reporting backbone of the prototype.

Data protection: After speaking with Tyler and Patrick from Xorbix, we implemented data masking in Databricks to help protect sensitive data. Their feedback pushed us to treat data protection as part of the working system, alongside capture, analysis, and reporting. Masking is one protective layer; it does not by itself establish full security or regulatory compliance.

Interface: Next.js, React, and TypeScript power a responsive dashboard deployed on Vercel. Databricks credentials stay server-side. A single recording control keeps the user experience simple while detailed evidence remains available in each session.

Challenges we ran into
The hardest model problem was the gap between hospital microphones and consumer headsets. In an early headset recording, speech was mistaken for snoring and sleep. We added AudioSet evidence gates and checked their effect on held-out recordings, improving snore F1 from 0.42 to 0.47 while changing fewer than 0.4% of predicted asleep/apnea seconds.

Hardware and infrastructure mattered just as much. We worked through Bluetooth microphone profiles, Raspberry Pi Wi-Fi compatibility, continuous chunk uploads, and stop-to-report orchestration. Repeated laptop GPU crashes forced us to finish training on CPU and keep production inference on Databricks serverless CPU.

Accomplishments we're proud of
Built the complete dashboard-to-Pi-to-Databricks-to-report path, including live audio uploads and remote Start/Stop control.
Verified an eight-minute reference recording through the Pi's actual capture and processing pipeline: 48 chunks uploaded, processing completed, and the report appeared on the dashboard.
Achieved apnea-plus-hypopnea event F1 of 0.835 on tracheal audio and 0.801 on ambient audio in the held-out 10-subject hospital dataset. These are offline dataset results, not clinical accuracy claims or headset validation.
Successfully paired the consumer Bluetooth headset and verified live microphone input. A full real-headset sleep validation remains future work.
Turned model output into both inspectable event evidence and a readable report.
Implemented Databricks data masking in response to direct feedback from Tyler and Patrick at Xorbix.
What we learned
A strong model is only one part of a useful health prototype. Microphone characteristics, recording quality, reliable uploads, honest uncertainty, and a clear explanation all determine whether a result is useful. We also learned to separate the jobs of the models: the audio model detects patterns; the language model explains structured findings rather than inventing a diagnosis.

Mentor conversations also changed our product decisions. Tyler and Patrick from Xorbix prompted an implemented data-protection improvement: masking. Carter from Strato VC encouraged us to focus the initial product on wellness rather than launch immediately as a medical device. That helped us separate a practical first business from the longer-term clinical vision. Wellness positioning must be reflected in actual functionality and claims; a label alone does not remove FDA obligations.

What's next for SleepSafe
Our first priority is paired headset and reference-recording evaluation, with particular attention to noise suppression, fit, false positives, and reduced-breathing events. We also want to improve signal-quality feedback and test the product with intended users and sleep-care professionals.

Wellness first: our Open Venture strategy
Following our conversation with Carter from Strato VC, our initial market direction is a wellness product that helps people understand their sleep patterns and nighttime sounds at home. This gives us a focused starting point for customer discovery and product development while we evaluate the more demanding evidence and regulatory requirements of a medical product.

Our initial customer hypothesis is people seeking greater awareness of their sleep. We plan to test a reusable recording kit and reporting service, establish hardware and cloud unit economics, and learn which insights users find useful. Our differentiation is the complete path from accessible audio capture to timestamped evidence and an understandable report. Our team has already built across embedded capture, audio machine learning, cloud workflows, and product design. Mentor feedback is an early signal shaping the strategy, not a claim of customer traction or investment.

Longer term: a clinician-directed overnight home assessment
We aim to use future wellness revenue to support clinical partnerships, validation studies, and development toward an appropriately FDA-cleared or approved medical device, depending on its intended use and applicable pathway. The ambition is to become a tool clinicians can order and interpret as part of care, much as they use other diagnostic tests; we are not claiming equivalence to a CT scan or any existing medical device.

The envisioned workflow is simple: a doctor lends or prescribes the device for one night, the patient sleeps with it at home, and returns it the next day. The clinician reviews the recording-derived findings to support diagnosis and decide on further testing or care. If validated and authorized for that use, this could reduce the need for some patients to spend a night in a sleep laboratory. It would not mean that every patient could avoid formal sleep testing.

We also intend to pursue insurance reimbursement to improve access and support a clinic-based business model. Coverage would require its own evidence and payer decisions; it is not guaranteed by FDA clearance or approval. Clinical partnerships, regulatory authorization, insurance coverage, and diagnostic use are future milestones, not capabilities or approvals we have today.

The earlier SleepSafe concept supplied the accessibility motivation. The implementation described here uses the current Raspberry Pi, PANNs/BiGRU, Databricks, and Next.js architecture; earlier slide-deck claims about motion sensors, diagnosing multiple disorders, and hardware pricing do not describe this prototype.
