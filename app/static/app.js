// Simple chat client for POST /ask. No dependencies, no build step.

const messages = document.getElementById("messages");
const welcome = document.getElementById("welcome");
const form = document.getElementById("ask-form");
const input = document.getElementById("query");
const submit = document.getElementById("submit");
const sidebar = document.getElementById("sidebar");

// Server-issued conversation ID; null until the first answer. Sending it
// back gives the agent the previous turns, so follow-ups work.
let conversationId = null;

// Append a labelled chat row ("You" / "Assistant" + bubble) and scroll it
// into view. Returns the bubble so the loading state can be updated in place.
function appendBubble(role, text) {
  welcome.hidden = true;

  const row = document.createElement("div");
  row.className = `row ${role}`;

  const meta = document.createElement("span");
  meta.className = "meta";
  meta.textContent = role === "user" ? "You" : "Assistant";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  row.append(meta, bubble);
  messages.appendChild(row);
  messages.scrollTop = messages.scrollHeight;
  return bubble;
}

// Build the collapsible "Details" section: symbol chips + raw response JSON.
function renderDetails(data) {
  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = "Details";
  details.appendChild(summary);

  if (data.tickers.length > 0) {
    const chips = document.createElement("div");
    chips.className = "chips";
    for (const ticker of data.tickers) {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = ticker;
      chips.appendChild(chip);
    }
    details.appendChild(chips);
  }

  const raw = document.createElement("pre");
  raw.textContent = JSON.stringify(data, null, 2);
  details.appendChild(raw);
  return details;
}

// Send one question: user bubble -> loading bubble -> answer or error.
async function sendMessage(query) {
  submit.disabled = true;
  appendBubble("user", query);
  const pending = appendBubble("assistant loading", "Thinking…");
  // pending's row carries "assistant loading"; resolved to a final state below

  try {
    const resp = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, conversation_id: conversationId }),
    });
    const data = await resp.json();

    if (!resp.ok) {
      const detail =
        typeof data.detail === "string" ? data.detail : "Request failed.";
      pending.parentElement.className = "row error";
      pending.textContent = detail;
      return;
    }

    conversationId = data.conversation_id;
    pending.parentElement.className = "row assistant";
    pending.textContent = data.answer;
    pending.appendChild(renderDetails(data));
  } catch (err) {
    pending.parentElement.className = "row error";
    pending.textContent = "Could not reach the server. Is it running?";
  } finally {
    submit.disabled = false;
    messages.scrollTop = messages.scrollHeight;
  }
}

function toggleSidebar() {
  sidebar.classList.toggle("open");
}

// --- event wiring ---------------------------------------------------------

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const query = input.value.trim();
  if (!query || submit.disabled) return;
  input.value = "";
  sendMessage(query);
});

// Enter sends; Shift+Enter inserts a newline
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

for (const button of document.querySelectorAll(".suggestion")) {
  button.addEventListener("click", () => {
    if (submit.disabled) return;
    sidebar.classList.remove("open"); // close overlay on mobile
    sendMessage(button.textContent);
  });
}

document.getElementById("sidebar-toggle").addEventListener("click", toggleSidebar);

document.getElementById("new-chat").addEventListener("click", () => {
  conversationId = null;
  messages.querySelectorAll(".row").forEach((row) => row.remove());
  welcome.hidden = false;
  sidebar.classList.remove("open");
});
