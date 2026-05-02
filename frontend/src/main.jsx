import React, { useMemo, useState, useEffect } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "https://stress-detector-api-yhx4.onrender.com";

// Whack-a-Mole Style Stress Buster Game
function StressBusterGame() {
  const [score, setScore] = useState(0);
  const [gameActive, setGameActive] = useState(false);
  const [timeLeft, setTimeLeft] = useState(30);
  const [targets, setTargets] = useState([]);
  const [gameOver, setGameOver] = useState(false);

  const emojis = ["😤", "😰", "😩", "🤯", "😡", "🧠"];

  // Game timer
  React.useEffect(() => {
    if (!gameActive || gameOver) return;
    
    const timer = setInterval(() => {
      setTimeLeft(prev => {
        if (prev <= 1) {
          setGameActive(false);
          setGameOver(true);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [gameActive, gameOver]);

  // Spawn targets
  React.useEffect(() => {
    if (!gameActive || gameOver) return;

    const spawn = setInterval(() => {
      const newTarget = {
        id: Math.random(),
        emoji: emojis[Math.floor(Math.random() * emojis.length)],
        top: Math.random() * 80,
        left: Math.random() * 90
      };
      setTargets(prev => [...prev, newTarget]);
    }, 600);

    return () => clearInterval(spawn);
  }, [gameActive, gameOver]);

  const startGame = () => {
    setGameActive(true);
    setGameOver(false);
    setScore(0);
    setTimeLeft(30);
    setTargets([]);
  };

  const hitTarget = (id) => {
    setTargets(prev => prev.filter(t => t.id !== id));
    setScore(prev => prev + 10);
  };

  const resetGame = () => {
    setScore(0);
    setTimeLeft(30);
    setGameActive(false);
    setGameOver(false);
    setTargets([]);
  };

  return (
    <div className="stress-game">
      <h3>🎯 Stress Buster! Click away your worries!</h3>
      <div className="game-header">
        <div className="game-score">Score: {score}</div>
        <div className="game-timer">Time: {timeLeft}s</div>
      </div>

      {!gameActive && !gameOver && (
        <button className="start-game-btn" onClick={startGame}>
          🎮 START GAME
        </button>
      )}

      {gameActive && (
        <div className="game-board-mole">
          {targets.map(target => (
            <div
              key={target.id}
              className="mole-target"
              style={{ top: `${target.top}%`, left: `${target.left}%` }}
              onClick={() => hitTarget(target.id)}
            >
              {target.emoji}
            </div>
          ))}
          <div className="game-instructions">Click on the stress emojis!</div>
        </div>
      )}

      {gameOver && (
        <div className="game-over-screen">
          <h4>🎉 Game Over!</h4>
          <p className="final-score">Final Score: {score}</p>
          <p className="game-message">
            {score > 150 ? "🔥 Stress destroyer! You're amazing!" : 
             score > 100 ? "💪 Great job busting that stress!" :
             "👍 Nice effort! Try again for a better score!"}
          </p>
          <button className="reset-game-btn" onClick={startGame}>
            Play Again
          </button>
          <button className="reset-game-btn secondary" onClick={resetGame}>
            Close Game
          </button>
        </div>
      )}
    </div>
  );
}

// Cute Animations Component
function CuteAnimations() {
  return (
    <div className="cute-animations">
      <div className="floating-emoji emoji-1">✨</div>
      <div className="floating-emoji emoji-2">🌸</div>
      <div className="floating-emoji emoji-3">🦋</div>
      <div className="floating-emoji emoji-4">🌟</div>
      <div className="bouncing-blob blob-1" />
      <div className="bouncing-blob blob-2" />
      <div className="bouncing-blob blob-3" />
    </div>
  );
}

const fieldGroups = [
  {
    title: "Lifestyle",
    fields: [
      { name: "sleep_hours", label: "Sleep hours", min: 2, max: 10, step: 0.5, value: 6 },
      { name: "screen_hours", label: "Screen hours", min: 0, max: 12, step: 0.5, value: 5 },
      { name: "exercise", label: "Exercise gap", min: 1, max: 4, step: 1, value: 2 },
      { name: "weight_change", label: "Weight change", min: 0, max: 3, step: 1, value: 0 }
    ]
  },
  {
    title: "Academics",
    fields: [
      { name: "cgpa", label: "CGPA", min: 0, max: 10, step: 0.1, value: 7.5 },
      { name: "study_load", label: "Study load", min: 1, max: 5, step: 1, value: 3 },
      { name: "attendance", label: "Attendance strain", min: 1, max: 4, step: 1, value: 2 },
      { name: "financial", label: "Financial pressure", min: 0, max: 4, step: 1, value: 1 }
    ]
  },
  {
    title: "Mind and Social",
    fields: [
      { name: "anxiety", label: "Anxiety", min: 1, max: 5, step: 1, value: 2 },
      { name: "depression_flag", label: "Low mood", min: 0, max: 4, step: 1, value: 1 },
      { name: "concentration", label: "Focus difficulty", min: 0, max: 4, step: 1, value: 1 },
      { name: "panic", label: "Panic frequency", min: 0, max: 4, step: 1, value: 0 },
      { name: "social_isolation", label: "Isolation", min: 1, max: 4, step: 1, value: 2 },
      { name: "peer_pressure", label: "Peer pressure", min: 1, max: 5, step: 1, value: 2 },
      { name: "home_stress", label: "Home stress", min: 1, max: 4, step: 1, value: 2 },
      { name: "relationship_stress", label: "Relationship stress", min: 0, max: 4, step: 1, value: 1 }
    ]
  }
];

const defaultInputs = fieldGroups
  .flatMap((group) => group.fields)
  .reduce((values, field) => ({ ...values, [field.name]: field.value }), {});

function App() {
  const [inputs, setInputs] = useState(defaultInputs);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const stressTone = useMemo(() => {
    const label = result?.stress_label?.toLowerCase();
    if (label === "high") return "high";
    if (label === "moderate") return "moderate";
    return "low";
  }, [result]);

  function updateField(name, value) {
    setInputs((current) => ({ ...current, [name]: Number(value) }));
  }

  async function predictStress(event) {
    event.preventDefault();
    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(inputs)
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Prediction failed");
      }

      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="workspace">
        <div className="intro">
          <p className="eyebrow">Riso check-in</p>
          <h1>Student Stress Detector</h1>
          <p>Bright, punchy student wellness signals with a print-studio pulse.</p>
          <div className="poster-strip" aria-hidden="true">
            <span>Sleep</span>
            <span>Study</span>
            <span>Mood</span>
            <span>Support</span>
          </div>
        </div>

        <form className="survey-panel" onSubmit={predictStress}>
          {fieldGroups.map((group) => (
            <fieldset key={group.title}>
              <legend>{group.title}</legend>
              <div className="field-grid">
                {group.fields.map((field) => (
                  <label className="field" key={field.name}>
                    <span>
                      {field.label}
                      <strong>{inputs[field.name]}</strong>
                    </span>
                    <input
                      type="range"
                      min={field.min}
                      max={field.max}
                      step={field.step}
                      value={inputs[field.name]}
                      onChange={(event) => updateField(field.name, event.target.value)}
                    />
                    <small>
                      {field.min} to {field.max}
                    </small>
                  </label>
                ))}
              </div>
            </fieldset>
          ))}

          <div className="actions">
            <button type="submit" disabled={loading}>
              {loading ? "Analyzing..." : "Predict stress"}
            </button>
            {error ? <p className="error">{error}</p> : null}
          </div>
        </form>
      </section>

      <aside className={`results-panel ${result ? stressTone : ""}`}>
        {result ? (
          <>
            <p className="eyebrow">Prediction</p>
            <div className="score-row">
              <div>
                <h2>{result.stress_label}</h2>
                <p>{result.confidence}% confidence</p>
              </div>
              <div className="score-ring" style={{ "--score": `${result.stress_pct}%` }}>
                <span>{result.stress_pct}%</span>
              </div>
            </div>

            <section>
              <h3>Probabilities</h3>
              {Object.entries(result.probabilities).map(([label, value]) => (
                <div className="bar-row" key={label}>
                  <span>{label}</span>
                  <div>
                    <i style={{ width: `${value}%` }} />
                  </div>
                  <strong>{value}%</strong>
                </div>
              ))}
            </section>

            <section>
              <h3>Factor Scores</h3>
              <div className="factor-grid">
                {Object.entries(result.factors).map(([label, value]) => (
                  <div className="factor" key={label}>
                    <span>{label}</span>
                    <strong>{value}%</strong>
                  </div>
                ))}
              </div>
            </section>

            <section>
              <h3>Recommended Actions</h3>
              <div className="solutions">
                {result.solutions.map((solution) => (
                  <article key={solution.title}>
                    <p>{solution.category}</p>
                    <h4>{solution.title}</h4>
                    <ul>
                      {solution.tips.slice(0, 3).map((tip) => (
                        <li key={tip}>{tip}</li>
                      ))}
                    </ul>
                    {solution.spotify_link && (
                      <a href={solution.spotify_link} target="_blank" rel="noopener noreferrer" className="spotify-btn">
                        🎵 Open on Spotify
                      </a>
                    )}
                  </article>
                ))}
              </div>
            </section>
          </>
        ) : (
          <div className="empty-state">
            <CuteAnimations />
            <p className="eyebrow">Ready</p>
            <h2>Run a survey to see the model response.</h2>
            <StressBusterGame />
          </div>
        )}
      </aside>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
