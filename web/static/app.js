const go = document.getElementById("go");
const note = document.getElementById("note");
const book = document.getElementById("book");
const controls = document.getElementById("controls");
const save = document.getElementById("save");
const pager = document.getElementById("pager");
const back = document.getElementById("back");
const forward = document.getElementById("forward");
const count = document.getElementById("count");

const pages = [];
const shown = new Set();
let at = -1;
let startedAt = 0;
let timer = null;

function clock() {
  const seconds = Math.max(0, Math.floor(Date.now() / 1000 - startedAt));
  note.className = "";
  note.innerHTML = "探索中 <b></b>";
  note.querySelector("b").textContent =
    `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

function turn(index, direction) {
  if (index < 0 || index >= pages.length || index === at) return;
  if (at >= 0) pages[at].hidden = true;
  at = index;
  const page = pages[at];
  page.hidden = false;
  page.className = "plate";
  void page.offsetWidth;
  page.classList.add(direction);
  save.href = `/api/image/${encodeURIComponent(page.dataset.image)}`;
  save.download = page.dataset.name;
  controls.hidden = false;
  count.textContent = `${at + 1} / ${pages.length}`;
  back.disabled = at === 0;
  forward.disabled = at === pages.length - 1;
  pager.hidden = pages.length < 2;
}

function place(image) {
  shown.add(image);
  const page = document.createElement("figure");
  page.className = "plate";
  page.hidden = true;
  page.dataset.image = image;
  page.dataset.name = image.replace(/^\d{8}-\d{6}-/, "");
  const plate = document.createElement("img");
  plate.src = `/api/image/${encodeURIComponent(image)}`;
  plate.alt = "観察された図版";
  page.append(plate);
  book.append(page);
  pages.push(page);
  turn(pages.length - 1, "turns-forward");
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
  if (state.image && !shown.has(state.image)) place(state.image);
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

back.addEventListener("click", () => turn(at - 1, "turns-back"));
forward.addEventListener("click", () => turn(at + 1, "turns-forward"));
document.addEventListener("keydown", (event) => {
  if (event.key === "ArrowLeft") turn(at - 1, "turns-back");
  if (event.key === "ArrowRight") turn(at + 1, "turns-forward");
});

poll();
setInterval(poll, 2000);
