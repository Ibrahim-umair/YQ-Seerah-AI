import { useState } from "react";
import "./App.css";

const suggestions = ["The first revelation", "Hijrah to Madinah", "Battle of Badr", "Life in Makkah"];

const demoResponse = {
  answer: [
    "The Year of Sorrow refers to the period in which the Prophet ﷺ experienced the passing of two of his greatest supporters—Khadijah (may Allah be pleased with her) and Abu Talib—within a short span of time.",
    "First, Sayyidah Khadijah passed away in the same year. She was the Prophet’s ﷺ first wife and believed in him from the very beginning, supporting him through Makkah’s early trials.",
    "Not long after, Abu Talib, his uncle and protector, also passed away. This left the Prophet ﷺ without the same protection from Quraysh.",
    "These back-to-back losses were a tremendous test, yet the Prophet ﷺ remained patient and continued his mission with unwavering trust in Allah.",
  ],
  primarySource: { episode: 17, timestamp: "34:21", timestampSeconds: 2061, videoId: "YOUR_YOUTUBE_VIDEO_ID" },
  otherSources: [
    { episode: 17, timestamp: "38:12" },
    { episode: 18, timestamp: "06:42" },
  ],
};

export default function App() {
  const [query, setQuery] = useState("");
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showSources, setShowSources] = useState(false);

  async function handleSubmit(e) {
    e?.preventDefault();
    const value = query.trim();
    if (!value || loading) return;
    setQuestion(value);
    setLoading(true);
    setShowSources(false);

    // Replace this demo block with your real RAG API call.
    setTimeout(() => {
      setResponse(demoResponse);
      setLoading(false);
    }, 700);
  }

  return (
    <div className="app">
      <header className="header">
        <a className="brand" href="/"><img src="/seerah-logo.png" alt="Seerah AI" /></a>
        <a href="#about" className="about-link">ⓘ About</a>
      </header>

      <main className="main">
        <section className="hero">
          <h1>Ask about the Seerah</h1>
          <div className="hero-ornament"><span /><b>✦</b><span /></div>
          <p>Answers sourced exclusively from<br /><strong>Dr. Yasir Qadhi’s Seerah lectures.</strong></p>

          <form className="search-box" onSubmit={handleSubmit}>
            <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="What would you like to learn about?" />
            <button type="submit" aria-label="Ask" disabled={loading}>↗</button>
          </form>

          {!response && (
            <div className="suggestions">
              <span>Try asking about:</span>
              <div>
                {suggestions.map((item) => <button key={item} type="button" onClick={() => setQuery(item)}>{item}</button>)}
              </div>
            </div>
          )}
        </section>

        {(question || loading) && (
          <section className="conversation">
            <div className="question-row">
              <div className="avatar">♙</div>
              <div className="question-bubble">{question}</div>
            </div>

            {loading ? (
              <div className="answer-row">
                <div className="avatar ai-avatar"><img src="/seerah-logo.png" alt="" /></div>
                <div className="answer-card loading"><span /><span /><span /></div>
              </div>
            ) : response && (
              <>
                <div className="answer-row">
                  <div className="avatar ai-avatar"><img src="/seerah-logo.png" alt="" /></div>
                  <article className="answer-card">
                    <div className="answer-content">{response.answer.map((p, i) => <p key={i}>{p}</p>)}</div>
                    <div className="feedback">Was this helpful? <button>♡</button><button>☹</button></div>
                  </article>
                </div>

                <div className="source-bar">
                  <div><small>PRIMARY SOURCE</small><strong>Episode {response.primarySource.episode} · {response.primarySource.timestamp}</strong></div>
                  <a target="_blank" rel="noreferrer" href={`https://www.youtube.com/watch?v=${response.primarySource.videoId}&t=${response.primarySource.timestampSeconds}s`}>Watch on YouTube ↗</a>
                  <button onClick={() => setShowSources(!showSources)}>{response.otherSources.length} more sources⌄</button>
                </div>

                {showSources && <div className="extra-sources">{response.otherSources.map((s, i) => <div key={i}><span>Episode {s.episode}</span><span>{s.timestamp}</span></div>)}</div>}

                <section className="video-card">
                  <div className="video-heading">Now playing from Episode {response.primarySource.episode} at <strong>{response.primarySource.timestamp}</strong></div>
                  <div className="video-wrap">
                    {response.primarySource.videoId !== "YOUR_YOUTUBE_VIDEO_ID" ? (
                      <iframe src={`https://www.youtube-nocookie.com/embed/${response.primarySource.videoId}?start=${response.primarySource.timestampSeconds}&rel=0`} title={`Seerah Episode ${response.primarySource.episode}`} allowFullScreen />
                    ) : (
                      <div className="video-placeholder"><span>✦</span><small>SEERAH OF THE</small><h2>Prophet Muhammad ﷺ</h2><p>Add your backend-returned YouTube video ID and this becomes the real player.</p></div>
                    )}
                  </div>
                </section>
              </>
            )}
          </section>
        )}

        {!response && !loading && (
          <section className="info-section" id="about">
            <div><small>ABOUT SEERAH AI</small><h2>Explore the Seerah through the original lectures.</h2></div>
            <p>Ask a question about the life of the Prophet ﷺ and receive an answer grounded in Yasir Qadhi’s Seerah series. Every answer points back to the relevant lecture and timestamp.</p>
          </section>
        )}
      </main>

      <footer><div>◈</div><p>Seerah AI can make mistakes. Please refer to the original lectures for full context.<br />This tool is not affiliated with Dr. Yasir Qadhi.</p></footer>
    </div>
  );
}
