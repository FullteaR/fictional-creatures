const go = document.getElementById("go");
const note = document.getElementById("note");
const plate = document.getElementById("plate");
const img = document.getElementById("img");

let startedAt = 0;
let timer = null;

function clock() {
  const seconds = Math.max(0, Math.floor(Date.now() / 1000 - startedAt));
  note.className = "";
  note.innerHTML = "探索中 <b></b>";
  note.querySelector("b").textContent =
    `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

function show(state) {
  go.disabled = state.running;
  if (state.running) {
    if (!timer) {
      startedAt = startedAt || Date.now() / 1000;
      timer = setInterval(clock, 1000);
    }
    clock();
    return;
  }
  clearInterval(timer);
  timer = null;
  startedAt = 0;
  if (state.error) {
    note.className = "is-alarm";
    note.textContent = `探索できませんでした — ${state.error}`;
  } else {
    note.className = "";
    note.textContent = "";
  }
  if (state.image) {
    const url = `/api/image/${encodeURIComponent(state.image)}`;
    if (img.getAttribute("src") !== url) {
      img.src = url;
      plate.hidden = false;
      plate.classList.remove("is-new");
      void plate.offsetWidth;
      plate.classList.add("is-new");
    }
  }
}

async function poll() {
  try {
    show(await fetch("/api/status").then((r) => r.json()));
  } catch {
    note.className = "is-alarm";
    note.textContent = "サーバーに繋がりません。docker compose up で webui を起動してください。";
  }
}

go.addEventListener("click", async () => {
  go.disabled = true;
  startedAt = Date.now() / 1000;
  const response = await fetch("/api/generate", { method: "POST" });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    note.className = "is-alarm";
    note.textContent = detail.detail || "探索を始められませんでした。";
  }
  poll();
});

poll();
setInterval(poll, 2000);
