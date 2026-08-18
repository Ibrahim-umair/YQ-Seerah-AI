import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL;

// DM Sans (loaded from Google Fonts) has no Arabic glyphs, so rendering ﷺ
// under it makes the browser probe Google's other unicode-range subsets
// looking for a match - one of those subset files 404s. Isolating the glyph
// in a span with a font stack that never names DM Sans stops the browser
// from asking Google Fonts about it at all; a local system font covers it
// instead, with no network request.
function withSafeGlyphs(text) {
  return text.split("ﷺ").flatMap((part, i, arr) =>
    i < arr.length - 1 ? [part, <span key={i} className="safe-glyph">ﷺ</span>] : [part]
  );
}

// Real questions from the eval set (data/eval_questions_raw.json), not
// invented - each one is grounded in a specific, verified incident rather
// than a vague topic, so retrieval lands on a narrow, precise set of chunks
// instead of scattering across everything tagged with a broad theme. Kept
// deliberately short (most under 100 characters) so they read well as compact
// suggestion chips; a few are cross-episode (T3), tying two arcs together.
const categories = [
  {
    name: "Before Revelation",
    prompts: [
      "How certain are we really about the exact day the Prophet was born?",
      "What did the Prophet do for a living as a young man, and what was he paid for it?",
      "How did Khadija رضي الله عنها, a wealthy woman who had turned down many suitors, end up marrying the Prophet?",
      "If the Arabs traced their religion back to Abraham, how did they end up worshipping idols at all?",
    ],
  },
  {
    name: "The Meccan Period",
    prompts: [
      "How did Hamza come to accept Islam?",
      "How did the number of daily prayers end up at five?",
      "When his uncle asked him to abandon his message, what was the Prophet's ﷺ reply?",
      "What price did Suhayb the Roman have to pay before the Quraysh would let him leave Mecca?",
      "Did Bilal ever come face to face with the man who used to torture him, and what happened?",
    ],
  },
  {
    name: "The Hijrah",
    prompts: [
      "Out of all the places the Muslims could have gone, why Madinah?",
      "What happened to Umm Salama رضي الله عنها when her husband tried to take the family out of Makkah?",
      "How did Suhayb al-Rumi get past the Quraysh when they came out to stop him leaving Makkah?",
      "Once the Quraysh realised the Prophet had slipped out of Makkah, what price did they put on him?",
      "What did the Quraysh decide to do about the Prophet at their late-night meeting, and who was kept out of it?",
    ],
  },
  {
    name: "Early Madinan Period",
    prompts: [
      "Who actually killed Abu Jahl, and how did he die?",
      "How did Abu Lahab die?",
      "Did the trench actually keep the enemy out of Medina?",
      "Was anyone actually punished for spreading the rumour about Aisha رضي الله عنها?",
      "Why had Salman al-Farisi never fought in a battle before the trench was dug?",
    ],
  },
  {
    name: "Treaties, Conquest & Final Years",
    prompts: [
      "Did the Muslims win at Mu'tah or lose?",
      "Who ended up holding the keys to the Kaaba after Mecca was taken?",
      "Why did the siege of Ta'if end without the city ever being taken?",
      "What happened to the man who killed Hamza, once the Muslims had the upper hand?",
      "What did the Persian emperor do when the letter from Muhammad ﷺ was delivered to him?",
    ],
  },
];

function getVideoId(youtubeUrl) {
  try {
    return new URL(youtubeUrl).searchParams.get("v");
  } catch {
    return null;
  }
}

// A bare YouTube iframe shows a blank/black frame for a couple of seconds
// while the embed page itself loads before it can paint anything, even
// just the thumbnail. Showing our own thumbnail image first - a single,
// instantly-loading file - and only mounting the real iframe once the user
// actually clicks play avoids that flash for anyone who never clicks, and
// moves it after an intentional action for anyone who does.
function VideoEmbed({ videoId, lectureNumber, startSeconds }) {
  const [playing, setPlaying] = useState(false);

  if (playing) {
    return (
      <iframe
        src={`https://www.youtube-nocookie.com/embed/${videoId}?start=${startSeconds}&autoplay=1&rel=0`}
        title={`Seerah Lecture ${lectureNumber}`}
        allow="autoplay; encrypted-media"
        allowFullScreen
      />
    );
  }
  return (
    <button type="button" className="video-facade" onClick={() => setPlaying(true)} aria-label="Play video">
      <img src={`https://i.ytimg.com/vi/${videoId}/hqdefault.jpg`} alt="" loading="lazy" />
      <span className="play-btn">▶</span>
    </button>
  );
}

function ArrowIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="12" y1="19" x2="12" y2="5" />
      <polyline points="5 12 12 5 19 12" />
    </svg>
  );
}

function BackIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
      <line x1="19" y1="12" x2="5" y2="12" />
      <polyline points="12 19 5 12 12 5" />
    </svg>
  );
}

function CloseIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="5" y1="5" x2="19" y2="19" />
      <line x1="19" y1="5" x2="5" y2="19" />
    </svg>
  );
}

function GithubIcon() {
  return (
    <svg width="19" height="19" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 .5C5.65.5.5 5.65.5 12c0 5.09 3.29 9.4 7.86 10.93.57.1.79-.25.79-.55 0-.27-.01-1.17-.02-2.12-3.2.7-3.88-1.36-3.88-1.36-.52-1.34-1.28-1.69-1.28-1.69-1.04-.72.08-.7.08-.7 1.15.08 1.76 1.18 1.76 1.18 1.03 1.75 2.7 1.25 3.36.96.1-.75.4-1.25.73-1.54-2.55-.29-5.24-1.28-5.24-5.68 0-1.25.45-2.28 1.18-3.08-.12-.29-.51-1.46.11-3.05 0 0 .96-.31 3.15 1.18a10.9 10.9 0 0 1 5.73 0c2.19-1.49 3.15-1.18 3.15-1.18.62 1.59.23 2.76.11 3.05.74.8 1.18 1.83 1.18 3.08 0 4.41-2.7 5.38-5.27 5.67.42.36.78 1.07.78 2.16 0 1.56-.01 2.82-.01 3.2 0 .3.21.66.8.55A11.5 11.5 0 0 0 23.5 12C23.5 5.65 18.35.5 12 .5Z" />
    </svg>
  );
}

function LinkedinIcon() {
  return (
    <svg width="19" height="19" viewBox="0 0 24 24" fill="currentColor">
      <path d="M20.45 20.45h-3.55v-5.57c0-1.33-.02-3.03-1.85-3.03-1.85 0-2.14 1.44-2.14 2.93v5.67H9.36V9h3.41v1.56h.05c.48-.9 1.63-1.85 3.36-1.85 3.59 0 4.26 2.37 4.26 5.45v6.29ZM5.34 7.43a2.06 2.06 0 1 1 0-4.12 2.06 2.06 0 0 1 0 4.12ZM7.12 20.45H3.56V9h3.56v11.45Z" />
    </svg>
  );
}

export default function App() {
  const [query, setQuery] = useState("");
  const [turns, setTurns] = useState([]);
  const [pendingQuestion, setPendingQuestion] = useState(null);
  const [previousResponseId, setPreviousResponseId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [statusText, setStatusText] = useState("");
  const [error, setError] = useState(null);
  const [selectedCategory, setSelectedCategory] = useState(null);
  const [showWhy, setShowWhy] = useState(false);
  const pendingRef = useRef(null);

  useEffect(() => {
    if (loading) pendingRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [loading]);

  useEffect(() => {
    if (!showWhy) return;
    const onKeyDown = (e) => e.key === "Escape" && setShowWhy(false);
    document.addEventListener("keydown", onKeyDown);
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.body.style.overflow = "";
    };
  }, [showWhy]);

  async function handleSubmit(e) {
    e?.preventDefault();
    const question = query.trim();
    if (!question || loading) return;

    setQuery("");
    setPendingQuestion(question);
    setLoading(true);
    setStatusText("");
    setError(null);

    // Set once the first "token" event arrives, so a "done" event that
    // shows up with no tokens before it (the rare case where the model
    // produces no streamed text at all) still creates the turn itself.
    let started = false;

    function appendToLastTurn(patch) {
      setTurns((prev) => {
        const next = [...prev];
        const last = next.length - 1;
        next[last] = typeof patch === "function" ? patch(next[last]) : { ...next[last], ...patch };
        return next;
      });
    }

    try {
      const res = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, previous_response_id: previousResponseId }),
      });
      if (!res.ok) throw new Error(`Server returned ${res.status}`);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const events = buffer.split("\n\n");
        buffer = events.pop(); // last piece may be an incomplete event - keep it for next read

        for (const raw of events) {
          if (!raw.startsWith("data: ")) continue;
          const evt = JSON.parse(raw.slice(6));

          if (evt.type === "status") {
            setStatusText(evt.text);
          } else if (evt.type === "token") {
            if (!started) {
              started = true;
              setPendingQuestion(null);
              setLoading(false);
              setStatusText("");
              setTurns((prev) => [
                ...prev,
                { id: crypto.randomUUID(), dbId: null, question, answer: evt.text,
                  sources: [], feedback: null, feedbackError: false,
                  timestampFeedback: null, timestampFeedbackError: false,
                  showAllSources: false, done: false },
              ]);
            } else {
              appendToLastTurn((t) => ({ ...t, answer: t.answer + evt.text }));
            }
          } else if (evt.type === "done") {
            if (!started) {
              setPendingQuestion(null);
              setLoading(false);
              setStatusText("");
              setTurns((prev) => [
                ...prev,
                { id: crypto.randomUUID(), dbId: evt.id, question, answer: evt.answer,
                  sources: evt.sources, feedback: null, feedbackError: false,
                  timestampFeedback: null, timestampFeedbackError: false,
                  showAllSources: false, done: true },
              ]);
            } else {
              appendToLastTurn({ dbId: evt.id, sources: evt.sources, done: true });
            }
            setPreviousResponseId(evt.response_id);
          } else if (evt.type === "error") {
            throw new Error(evt.message);
          }
        }
      }
    } catch {
      setError("Something went wrong reaching the Seerah AI - please try again.");
    } finally {
      setPendingQuestion(null);
      setLoading(false);
      setStatusText("");
    }
  }

  function startNewChat() {
    setTurns([]);
    setPreviousResponseId(null);
    setQuery("");
    setError(null);
    setSelectedCategory(null);
  }

  function toggleSources(turnId) {
    setTurns((prev) => prev.map((t) => (t.id === turnId ? { ...t, showAllSources: !t.showAllSources } : t)));
  }

  async function sendFeedback(turnId, dbId, score) {
    setTurns((prev) => prev.map((t) => (t.id === turnId ? { ...t, feedbackError: false } : t)));
    try {
      const res = await fetch(`${API_URL}/feedback/${dbId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ score }),
      });
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      setTurns((prev) => prev.map((t) => (t.id === turnId ? { ...t, feedback: score } : t)));
    } catch {
      setTurns((prev) => prev.map((t) => (t.id === turnId ? { ...t, feedbackError: true } : t)));
    }
  }

  async function sendTimestampFeedback(turnId, dbId, score) {
    setTurns((prev) => prev.map((t) => (t.id === turnId ? { ...t, timestampFeedbackError: false } : t)));
    try {
      const res = await fetch(`${API_URL}/feedback/${dbId}/timestamp`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ score }),
      });
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      setTurns((prev) => prev.map((t) => (t.id === turnId ? { ...t, timestampFeedback: score } : t)));
    } catch {
      setTurns((prev) => prev.map((t) => (t.id === turnId ? { ...t, timestampFeedbackError: true } : t)));
    }
  }

  const hasConversation = turns.length > 0 || loading;

  return (
    <div className="app">
      <header className="header">
        <a className="brand" href="/"><img src="/seerah-logo.png" alt="Seerah AI" /></a>
        <button type="button" className="about-link" onClick={() => setShowWhy(true)}>About</button>
      </header>

      <main className="main">
        <section className="hero">
          <h1>Ask about the Seerah</h1>
          <div className="hero-ornament"><span /><b>✦</b><span /></div>
          <p>Answers sourced exclusively from<br /><strong>Dr. Yasir Qadhi's Seerah lectures.</strong></p>

          <form className="search-box" onSubmit={handleSubmit}>
            <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="What would you like to learn about?" />
            {hasConversation && (
              <button type="button" className="clear-chat-btn" onClick={startNewChat} aria-label="Clear chat" title="Clear chat">
                <img src="/cleaning-broom-2.png" alt="" />
              </button>
            )}
            <button type="submit" aria-label="Ask" disabled={loading}><ArrowIcon /></button>
          </form>

          {turns.length > 0 && !loading && turns[turns.length - 1].done && (
            <p className="new-topic-hint">Got an unrelated question? Clear this conversation first for the best answer.</p>
          )}

          {!hasConversation && (
            <div className="suggestions">
              {!selectedCategory ? (
                <>
                  <span>Explore by period:</span>
                  <div>
                    {categories.map((cat) => (
                      <button key={cat.name} type="button" onClick={() => setSelectedCategory(cat.name)}>
                        {cat.name}
                      </button>
                    ))}
                  </div>
                </>
              ) : (
                <>
                  <span className="category-label">
                    <button type="button" className="back-icon-btn" aria-label="Back to categories" onClick={() => setSelectedCategory(null)}>
                      <BackIcon />
                    </button>
                    {selectedCategory}
                  </span>
                  <div>
                    {categories.find((c) => c.name === selectedCategory).prompts.map((item) => (
                      <button key={item} type="button" onClick={() => setQuery(item)}>{withSafeGlyphs(item)}</button>
                    ))}
                  </div>
                </>
              )}
            </div>
          )}
        </section>

        {error && <div className="error-banner">{error}</div>}

        {hasConversation && (
          <section className="conversation">
            {turns.map((turn) => {
              const [primary, ...others] = turn.sources;
              const videoId = primary ? getVideoId(primary.youtube_url) : null;
              return (
                <div key={turn.id} className="turn">
                  <div className="question-row">
                    <div className="avatar">♙</div>
                    <div className="question-bubble">{turn.question}</div>
                  </div>

                  <div className="answer-row">
                    <div className="avatar ai-avatar"><img src="/seerah-logo.png" alt="" /></div>
                    <article className="answer-card">
                      <div className="answer-content">
                        <ReactMarkdown>{turn.answer}</ReactMarkdown>
                      </div>
                      {turn.done && (
                        <div className="feedback">
                          Was this helpful?
                          <button
                            className={turn.feedback === 1 ? "selected" : ""}
                            disabled={turn.feedback !== null}
                            onClick={() => sendFeedback(turn.id, turn.dbId, 1)}
                            aria-label="Helpful"
                          ><span>👍</span></button>
                          <button
                            className={turn.feedback === -1 ? "selected" : ""}
                            disabled={turn.feedback !== null}
                            onClick={() => sendFeedback(turn.id, turn.dbId, -1)}
                            aria-label="Not helpful"
                          ><span>👎</span></button>
                          {turn.feedbackError && <small className="feedback-error">couldn't save, try again</small>}
                        </div>
                      )}
                    </article>
                  </div>

                  {!turn.done && (
                    <>
                      <div className="source-bar source-bar-loading"><span /><span /></div>
                      <section className="video-card">
                        <div className="video-heading">Generating video reference&hellip;</div>
                        <div className="video-wrap video-wrap-loading"><span /></div>
                      </section>
                    </>
                  )}

                  {primary && (
                    <>
                      <div className="source-bar">
                        <div>
                          <small>PRIMARY SOURCE</small>
                          <strong>Lecture {primary.lecture_number} &middot; {primary.start_timestamp}</strong>
                        </div>
                        <a target="_blank" rel="noreferrer" href={primary.timestamped_url}>Watch on YouTube ↗</a>
                        {others.length > 0 && (
                          <button onClick={() => toggleSources(turn.id)}>{others.length} more source{others.length > 1 ? "s" : ""}⌄</button>
                        )}
                      </div>

                      {turn.showAllSources && (
                        <div className="extra-sources">
                          {others.map((s, i) => (
                            <a key={i} href={s.timestamped_url} target="_blank" rel="noreferrer">
                              <span>Lecture {s.lecture_number}</span><span>{s.start_timestamp}</span>
                            </a>
                          ))}
                        </div>
                      )}

                      <section className="video-card">
                        <div className="video-heading">
                          Now playing from Lecture {primary.lecture_number} at <strong>{primary.start_timestamp}</strong>
                        </div>
                        <div className="video-wrap">
                          {videoId ? (
                            <VideoEmbed
                              videoId={videoId}
                              lectureNumber={primary.lecture_number}
                              startSeconds={Math.floor(primary.start_timestamp_seconds)}
                            />
                          ) : (
                            <div className="video-placeholder"><span>✦</span><small>SEERAH OF THE</small><h2>{withSafeGlyphs("Prophet Muhammad ﷺ")}</h2></div>
                          )}
                        </div>
                        <div className="timestamp-feedback">
                          Right moment?
                          <button
                            className={turn.timestampFeedback === 1 ? "selected" : ""}
                            disabled={turn.timestampFeedback !== null}
                            onClick={() => sendTimestampFeedback(turn.id, turn.dbId, 1)}
                            aria-label="Timestamp correct"
                          ><span>👍</span></button>
                          <button
                            className={turn.timestampFeedback === -1 ? "selected" : ""}
                            disabled={turn.timestampFeedback !== null}
                            onClick={() => sendTimestampFeedback(turn.id, turn.dbId, -1)}
                            aria-label="Timestamp incorrect"
                          ><span>👎</span></button>
                          {turn.timestampFeedbackError && <small className="feedback-error">couldn't save, try again</small>}
                        </div>
                      </section>
                    </>
                  )}
                </div>
              );
            })}

            {loading && (
              <div className="turn" ref={pendingRef}>
                <div className="question-row">
                  <div className="avatar">♙</div>
                  <div className="question-bubble">{pendingQuestion}</div>
                </div>
                <div className="answer-row">
                  <div className="avatar ai-avatar"><img src="/seerah-logo.png" alt="" /></div>
                  <div className="answer-card loading">
                    {statusText && <p className="status-line">{statusText}</p>}
                    <div className="shimmer-bars"><span /><span /><span /></div>
                  </div>
                </div>
              </div>
            )}
          </section>
        )}

        {!hasConversation && (
          <section className="info-section" id="about">
            <div><small>ABOUT SEERAH AI</small><h2>Explore the Seerah through the original lectures.</h2></div>
            <p>{withSafeGlyphs("Ask a question about the life of the Prophet ﷺ and receive an answer grounded in Yasir Qadhi's Seerah series. Every answer points back to the relevant lecture and timestamp.")}</p>
          </section>
        )}
      </main>

      <footer>
        <p className="footer-primary">Seerah AI can make mistakes &mdash; refer to the original lectures for full context.</p>
        <p className="footer-secondary">Not affiliated with Dr. Yasir Qadhi</p>
      </footer>

      {showWhy && (
        <div className="why-overlay" role="dialog" aria-modal="true" onClick={() => setShowWhy(false)}>
          <div className="why-card" onClick={(e) => e.stopPropagation()}>
            <button type="button" className="why-close" onClick={() => setShowWhy(false)} aria-label="Close">
              <CloseIcon />
            </button>

            <div className="why-zigzag why-zigzag-top" aria-hidden="true" />
            <p className="why-note">
              Shaykh Yasir Qadhi's Seerah series is an incredible 150+ hour resource, but I constantly struggled to
              remember specific events, names, or lessons after listening. Finding a single point again took way
              too much time skipping through videos. I built Seerah AI to make the entire series easily
              searchable: just ask a question in plain English, and it takes you straight to the exact lecture and
              timestamp.
            </p>
            <div className="why-zigzag why-zigzag-bottom" aria-hidden="true" />

            <div className="why-thanks">
              <p>With thanks to Shaykh Dr. Yasir Qadhi, whose lectures are the foundation of everything here.</p>
              <p>
                And to{" "}
                <a href="https://www.linkedin.com/in/wasifmasood/" target="_blank" rel="noreferrer">Wasif Masood</a>,
                whose Hugging Face dataset of the lecture transcripts made this project possible.
              </p>
            </div>

            <div className="why-socials">
              <a href="https://github.com/Ibrahim-umair" target="_blank" rel="noreferrer" aria-label="GitHub"><GithubIcon /></a>
              <a href="https://www.linkedin.com/in/ibrahim-bin-umair-a99899247/" target="_blank" rel="noreferrer" aria-label="LinkedIn"><LinkedinIcon /></a>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
