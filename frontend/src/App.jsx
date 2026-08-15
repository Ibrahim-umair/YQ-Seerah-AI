import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL;

// Real questions from the eval set (data/eval_questions_raw.json), not
// invented - each one is grounded in a specific, verified incident rather
// than a vague topic, so retrieval lands on a narrow, precise set of chunks
// instead of scattering across everything tagged with a broad theme. Most
// are single-lecture (T1/T2); a few are deliberately cross-episode (T3),
// tying two arcs together, since those make for the most compelling prompts.
const categories = [
  {
    name: "Before Revelation",
    prompts: [
      "The well of Zamzam had been lost for generations. How was it found again, and what trouble did that stir up?",
      "Something happened while the Prophet was being raised out in the desert that frightened his foster mother into sending him back. What was it?",
      "A trader from Yemen was cheated out of his money in Mecca and had no clan there to back him. What did he do about it?",
      "Khadija was wealthy and had turned down plenty of suitors. How did she end up marrying the Prophet?",
      "The Prophet's grandfather wasn't actually born with the name everyone knows him by. How did he end up with it?",
    ],
  },
  {
    name: "The Meccan Period",
    prompts: [
      "After that first encounter in the cave, nothing more came for a while. What happened during that gap and how did it end?",
      "When the Negus demanded to know what the Muslim refugees believed about Jesus, what answer did Ja'far give, and how did the king react?",
      "After the people of Ta'if drove the Prophet out, he was offered the chance to have them destroyed. Who made the offer and what did he say?",
      "On the night journey the Prophet was offered a choice between two drinks. Which did he take, and what was he told about his choice?",
      "Was the king of Abyssinia already a Muslim when he gave the emigrants refuge, or did that come later?",
    ],
  },
  {
    name: "The Hijrah",
    prompts: [
      "How did Suhayb al-Rumi get past the Quraysh when they came out to stop him leaving Makkah?",
      "What happened to Umm Salama when her husband tried to take the family out of Makkah?",
      "On the road north a passing caravan recognised Abu Bakr and asked who the other traveller was. What did he tell them?",
      "A man caught up with the Prophet and Abu Bakr in the desert hoping to collect the reward. What became of him?",
      "The bedouin who chased the Prophet and Abu Bakr for the bounty during the migration turned up again at the Farewell Pilgrimage. What did he want to know there?",
    ],
  },
  {
    name: "Early Madinan Period",
    prompts: [
      "Who actually killed Abu Jahl, and how did he die?",
      "What became of the man who killed Hamza? Did he ever accept Islam, and how did the Prophet deal with him afterwards?",
      "When the Prophet got separated from the bulk of his army at Uhud, who was still with him, and how did that small group keep him alive?",
      "How did the Muslims find out that the Banu Qurayza had gone over to the enemy while Medina was surrounded?",
      "One man slipped out of the besieged fortress of Banu Qurayza at night and was let go instead of being killed. Who was he and why was he spared?",
    ],
  },
  {
    name: "Treaties, Conquest & Final Years",
    prompts: [
      "During the talks outside Mecca a Quraysh negotiator kept reaching for the Prophet's beard. Who stopped him?",
      "Who ended up holding the keys to the Kaaba after Mecca was taken?",
      "What did the Prophet give Ka'b ibn Zuhayr after hearing him recite his poem, and what did he ask of him afterwards?",
      "Umar would not accept the news that the Prophet had died. What was he doing in the mosque, and what finally brought him round?",
      "The Quraysh chief who stonewalled the Prophet over the treaty at Hudaybiyyah had been a Muslim prisoner years earlier. Where was he held, and what happened to him when Makkah fell?",
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

export default function App() {
  const [query, setQuery] = useState("");
  const [turns, setTurns] = useState([]);
  const [pendingQuestion, setPendingQuestion] = useState(null);
  const [previousResponseId, setPreviousResponseId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedCategory, setSelectedCategory] = useState(null);
  const pendingRef = useRef(null);

  useEffect(() => {
    if (loading) pendingRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [loading]);

  async function handleSubmit(e) {
    e?.preventDefault();
    const question = query.trim();
    if (!question || loading) return;

    setQuery("");
    setPendingQuestion(question);
    setLoading(true);
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

          if (evt.type === "token") {
            if (!started) {
              started = true;
              setPendingQuestion(null);
              setLoading(false);
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
        <a href="#about" className="about-link">About</a>
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
                      <button key={item} type="button" onClick={() => setQuery(item)}>{item}</button>
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
                            <iframe
                              src={`https://www.youtube-nocookie.com/embed/${videoId}?start=${Math.floor(primary.start_timestamp_seconds)}&rel=0`}
                              title={`Seerah Lecture ${primary.lecture_number}`}
                              allowFullScreen
                            />
                          ) : (
                            <div className="video-placeholder"><span>✦</span><small>SEERAH OF THE</small><h2>Prophet Muhammad ﷺ</h2></div>
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
                  <div className="answer-card loading"><span /><span /><span /></div>
                </div>
              </div>
            )}
          </section>
        )}

        {!hasConversation && (
          <section className="info-section" id="about">
            <div><small>ABOUT SEERAH AI</small><h2>Explore the Seerah through the original lectures.</h2></div>
            <p>Ask a question about the life of the Prophet ﷺ and receive an answer grounded in Yasir Qadhi's Seerah series. Every answer points back to the relevant lecture and timestamp.</p>
          </section>
        )}
      </main>

      <footer>
        <p className="footer-primary">Seerah AI can make mistakes &mdash; refer to the original lectures for full context.</p>
        <p className="footer-secondary">Not affiliated with Dr. Yasir Qadhi</p>
      </footer>
    </div>
  );
}
