import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL;

const suggestions = ["The first revelation", "Hijrah to Madinah", "Battle of Badr", "Life in Makkah"];

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

function BroomIcon() {
  return (
    <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
      <path d="M19 3 10 12" />
      <path d="M10 12 4 21" />
      <path d="M10 12 7 21.5" />
      <path d="M10 12 9.5 22" />
      <path d="M10 12 12 20.5" />
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

    try {
      const res = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, previous_response_id: previousResponseId }),
      });
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const data = await res.json();

      setTurns((prev) => [
        ...prev,
        { id: data.id, question, answer: data.answer, sources: data.sources, feedback: null, showAllSources: false },
      ]);
      setPreviousResponseId(data.response_id);
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
  }

  function toggleSources(turnId) {
    setTurns((prev) => prev.map((t) => (t.id === turnId ? { ...t, showAllSources: !t.showAllSources } : t)));
  }

  async function sendFeedback(turnId, score) {
    setTurns((prev) => prev.map((t) => (t.id === turnId ? { ...t, feedbackError: false } : t)));
    try {
      const res = await fetch(`${API_URL}/feedback/${turnId}`, {
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
                <BroomIcon />
              </button>
            )}
            <button type="submit" aria-label="Ask" disabled={loading}><ArrowIcon /></button>
          </form>

          {!hasConversation && (
            <div className="suggestions">
              <span>Try asking about:</span>
              <div>
                {suggestions.map((item) => <button key={item} type="button" onClick={() => setQuery(item)}>{item}</button>)}
              </div>
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
                      <div className="feedback">
                        Was this helpful?
                        <button
                          className={turn.feedback === 1 ? "selected" : ""}
                          disabled={turn.feedback !== null}
                          onClick={() => sendFeedback(turn.id, 1)}
                          aria-label="Helpful"
                        >👍</button>
                        <button
                          className={turn.feedback === -1 ? "selected" : ""}
                          disabled={turn.feedback !== null}
                          onClick={() => sendFeedback(turn.id, -1)}
                          aria-label="Not helpful"
                        >👎</button>
                        {turn.feedbackError && <small className="feedback-error">couldn't save, try again</small>}
                      </div>
                    </article>
                  </div>

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
