"""File: desktop/application.py
Purpose: Native custom-vector authoring, real runtime comparison and model-evidence review.
Symbols/variables: see docs/CODE_INDEX.md; Application retains its worker and profile through Quit.
Prepared patterns are editable data in this complete workbench, not a second demonstration app.
"""

from __future__ import annotations

import csv
import io
import json
import queue
import threading
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any

from desktop.project import MAX_CSV, Workspace, parse_vectors, sample_text, verify_bundle
from desktop.storage import VERSION, new_json, read_regular
from edgevision.constants import CLASS_NAMES, IMAGE_SIDE
from edgevision.review import ReviewLimits

MAX_PREVIEW = 60_000
RUNTIME_NAMES = {
    "portable_numpy": "NumPy",
    "onnx_float32": "ONNX float32",
    "onnx_int8_dynamic_weights": "ONNX int8",
    "cpp": "C++",
    "rust": "Rust",
}


class Application:
    """One offline native window over the actual immutable models and retained
    scientific/native work."""

    def __init__(self, root: tk.Tk, workspace: Workspace):
        """Create actual author/review controls and keep all Tk mutation on its owning UI thread."""
        self.root, self.workspace = root, workspace
        self.busy = self.closing = self.closed = False
        self.worker: threading.Thread | None = None
        self.results: queue.Queue[tuple[bool, Any]] = queue.Queue()
        self.on_result: Callable[[Any], None] | None = None
        self.controls: list[Any] = []
        self.last_evidence = workspace.last_report()
        self.status = tk.StringVar(
            value="Ready. Prepare or import vectors; review the synthetic model boundary."
        )
        self.label = tk.StringVar(value="vertical")
        self.pixels = [[0.0] * 8 for _ in range(8)]
        self.gates = [tk.StringVar(value=value) for value in ("0.90", "0.75", "0.25", "0.01")]
        root.title(f"Edge Vision Workbench {VERSION} — Jason Stys")
        root.geometry("1260x880")
        root.minsize(1040, 740)
        root.protocol("WM_DELETE_WINDOW", self.close)
        shell = ttk.Frame(root, padding=16)
        shell.pack(fill="both", expand=True)
        ttk.Label(
            shell, text="Edge Vision Deployment Workbench", font=("TkDefaultFont", 21, "bold")
        ).pack(anchor="w")
        ttk.Label(
            shell, text=f"Jason Stys · v{VERSION} · actual NumPy / ONNX / C++ / Rust engines"
        ).pack(anchor="w", pady=5)
        ttk.Label(
            shell,
            text=(
                "Synthetic 8x8 vertical/horizontal/diagonal shapes only. No camera, person "
                "recognition or deployment authority."
            ),
            wraplength=1160,
        ).pack(anchor="w", pady=(0, 10))
        footer = ttk.Frame(shell)
        footer.pack(side="bottom", fill="x", pady=(10, 0))
        ttk.Label(footer, textvariable=self.status, wraplength=960).pack(
            side="left", fill="x", expand=True
        )
        self.quit = ttk.Button(footer, text="Quit application", command=self.close)
        self.quit.pack(side="right")
        navigation = ttk.Frame(shell)
        navigation.pack(fill="x", pady=(0, 8))
        self._button(navigation, "Help & privacy", self.help)
        self.export = self._button(navigation, "Export captured evidence", self.export_result)
        self._button(navigation, "Inspect complete candidate", self.inspect)
        self._button(navigation, "Rebuild reproducible reference", self.rebuild)
        self.tabs = ttk.Notebook(shell)
        self.tabs.pack(fill="both", expand=True)
        custom, review, evidence = (ttk.Frame(self.tabs, padding=10) for _ in range(3))
        for frame, title in (
            (custom, "Custom vectors & real inference"),
            (review, "Deployment review gates"),
            (evidence, "Captured evidence"),
        ):
            self.tabs.add(frame, text=title)
        self._custom(custom)
        self._review(review)
        self.evidence = ScrolledText(evidence, wrap="word", font=("Consolas", 10))
        self.evidence.pack(fill="both", expand=True)
        self.evidence.configure(state="disabled")
        if self.last_evidence is not None:
            self.show_evidence(self.last_evidence)
        self._paint()
        root.after(30, self._poll)

    def _button(self, parent, title: str, command: Callable[[], None]) -> ttk.Button:
        """Register a normal action so pending work disables all competing controls."""
        button = ttk.Button(parent, text=title, command=command)
        button.pack(side="left", padx=(0, 7))
        self.controls.append(button)
        return button

    def _custom(self, frame) -> None:
        """Offer both exact CSV authoring/import and an editable 8x8 pixel surface."""
        actions = ttk.Frame(frame)
        actions.pack(fill="x")
        self.prepare = self._button(actions, "Prepare held-out sample data", self.prepare_data)
        self._button(actions, "Import vector CSV", self.import_vectors)
        self._button(actions, "Save vector CSV", self.save_vectors)
        self.python = self._button(actions, "Run NumPy + ONNX variants", lambda: self.infer(False))
        self.native = self._button(
            actions, "Compare all five actual engines", lambda: self.infer(True)
        )
        ttk.Label(
            frame,
            text=(
                "CSV: id, label, f0 … f63. Exactly 64 finite pixels in [0,1]; labels are your"
                " assertions. Maximum 256 rows / 1 MiB."
            ),
            wraplength=1120,
        ).pack(anchor="w", pady=(8, 4))
        vector_frame = ttk.Frame(frame)
        vector_frame.pack(fill="x")
        self.vectors = ScrolledText(vector_frame, height=9, wrap="none", font=("Consolas", 9))
        self.vectors.pack(fill="x")
        horizontal = ttk.Scrollbar(vector_frame, orient="horizontal", command=self.vectors.xview)
        horizontal.pack(fill="x")
        self.vectors.configure(xscrollcommand=horizontal.set)
        self.controls.append(self.vectors)
        lower = ttk.Frame(frame)
        lower.pack(fill="both", expand=True, pady=(10, 0))
        drawing = ttk.Frame(lower)
        drawing.pack(side="left", fill="y", padx=(0, 16))
        ttk.Label(drawing, text="Draw one synthetic 8x8 pattern").pack(anchor="w")
        self.canvas = tk.Canvas(
            drawing, width=208, height=208, background="white", highlightthickness=1
        )
        self.canvas.pack(pady=5)
        self.canvas.bind("<Button-1>", self._pixel)
        self.expected = ttk.Combobox(
            drawing, textvariable=self.label, values=CLASS_NAMES, state="readonly", width=22
        )
        self.expected.pack(fill="x")
        self.controls.append(self.expected)
        actions = ttk.Frame(drawing)
        actions.pack(pady=6)
        self._button(actions, "Use drawn vector", self.use_drawing)
        self._button(actions, "Clear drawing", self.clear_drawing)
        table = ttk.Frame(lower)
        table.pack(side="left", fill="both", expand=True)
        self.predictions = ttk.Treeview(
            table,
            columns=("runtime", "row", "class", "asserted", "probability", "agreement"),
            show="headings",
            height=10,
        )
        for name, width in (
            ("runtime", 160),
            ("row", 100),
            ("class", 100),
            ("asserted", 100),
            ("probability", 105),
            ("agreement", 115),
        ):
            self.predictions.heading(
                name,
                text={"asserted": "User label", "agreement": "Max abs. error"}.get(
                    name, name.replace("_", " ").title()
                ),
            )
            self.predictions.column(name, width=width, minwidth=75)
        self.predictions.pack(side="left", fill="both", expand=True)
        vertical = ttk.Scrollbar(table, orient="vertical", command=self.predictions.yview)
        vertical.pack(side="right", fill="y")
        self.predictions.configure(yscrollcommand=vertical.set)

    def _review(self, frame) -> None:
        """Make all configurable review thresholds explicit; original release gates stay
        unchanged."""
        ttk.Label(
            frame,
            text=(
                "Review every model on the same 48 held-out samples under clean, Gaussian "
                "noise, occlusion and dimming. Eligibility is advisory."
            ),
            wraplength=1120,
        ).pack(anchor="w")
        gates = ttk.Frame(frame)
        gates.pack(fill="x", pady=10)
        for index, title in enumerate(
            ("Minimum accuracy", "Minimum class recall", "Maximum ECE", "Maximum reference error")
        ):
            cell = ttk.Frame(gates)
            cell.pack(side="left", padx=(0, 16))
            ttk.Label(cell, text=title).pack(anchor="w")
            entry = ttk.Entry(cell, textvariable=self.gates[index], width=16)
            entry.pack()
            self.controls.append(entry)
        actions = ttk.Frame(frame)
        actions.pack(fill="x", pady=(0, 10))
        self.review_button = self._button(actions, "Run actual review envelope", self.review)
        self.review_rows = ttk.Treeview(
            frame,
            columns=("runtime", "condition", "accuracy", "recall", "ece", "error", "gates"),
            show="headings",
            height=15,
        )
        for name in ("runtime", "condition", "accuracy", "recall", "ece", "error", "gates"):
            self.review_rows.heading(name, text="ECE" if name == "ece" else name.title())
            self.review_rows.column(name, width=155 if name == "runtime" else 105, minwidth=80)
        self.review_rows.pack(fill="both", expand=True)

    def _paint(self) -> None:
        """Render only the64 owned authoring cells, with no downloaded images/fonts."""
        self.canvas.delete("all")
        for row in range(8):
            for column in range(8):
                shade = round(255 * (1 - self.pixels[row][column]))
                self.canvas.create_rectangle(
                    column * 26,
                    row * 26,
                    (column + 1) * 26,
                    (row + 1) * 26,
                    fill=f"#{shade:02x}{shade:02x}{shade:02x}",
                    outline="#9ba7ac",
                )

    def _pixel(self, event) -> None:
        """Toggle an authored bounded cell only while the application is ready."""
        if self.busy or self.closing:
            return
        row, column = event.y // 26, event.x // 26
        if 0 <= row < IMAGE_SIDE and 0 <= column < IMAGE_SIDE:
            self.pixels[row][column] = 0.0 if self.pixels[row][column] else 1.0
            self._paint()

    def use_drawing(self) -> None:
        """Turn the actual drawing into editable standard CSV; the chosen label is not
        inferred truth."""
        if self.busy or self.closing:
            return
        output = io.StringIO(newline="")
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["id", "label", *(f"f{index}" for index in range(64))])
        writer.writerow(
            ["drawn-pattern", self.label.get(), *(value for row in self.pixels for value in row)]
        )
        self._set_vectors(output.getvalue())
        self.status.set(
            "Drawing copied to editable CSV. Run inference; the selected label remains "
            "your assertion."
        )

    def clear_drawing(self) -> None:
        """Clear only authoring cells; saved vectors and captured results remain intact."""
        if self.busy or self.closing:
            return
        self.pixels = [[0.0] * IMAGE_SIDE for _ in range(IMAGE_SIDE)]
        self._paint()
        self.status.set("Drawing cleared. Saved vectors and captured evidence are unchanged.")

    def _set_vectors(self, text: str) -> None:
        """Replace only the editor after bounded validation; stored model/data are unchanged."""
        parse_vectors(text)
        self.vectors.delete("1.0", "end")
        self.vectors.insert("1.0", text)

    def prepare_data(self) -> None:
        """Prepare actual held-out dataset rows within the normal application editor."""
        if not self.busy and not self.closing:
            try:
                self._set_vectors(sample_text(self.workspace.active))
                self.status.set(
                    "Six actual held-out synthetic rows prepared. Edit, inspect or infer them."
                )
            except (OSError, ValueError) as error:
                self.status.set(f"Preparation refused: {error}")

    def import_vectors(self) -> None:
        """Use an explicit file picker and bounded descriptor read; never execute a
        selected file."""
        if self.busy or self.closing:
            return
        name = filedialog.askopenfilename(filetypes=[("Vector CSV", "*.csv")], parent=self.root)
        if not name:
            return
        try:
            self._set_vectors(read_regular(Path(name), MAX_CSV).decode("utf-8"))
            self.status.set("CSV imported into the editor; inference has not run yet.")
        except (OSError, ValueError) as error:
            self.status.set(f"Import refused: {error}")

    def save_vectors(self) -> None:
        """Save a valid authoring snapshot to a new CSV, retaining an existing user's file."""
        if self.busy or self.closing:
            return
        text = self.vectors.get("1.0", "end-1c")
        try:
            parse_vectors(text)
            name = filedialog.asksaveasfilename(
                defaultextension=".csv", filetypes=[("Vector CSV", "*.csv")], parent=self.root
            )
            if not name:
                return
            with Path(name).open("x", encoding="utf-8", newline="") as output:
                output.write(text)
                output.flush()
            self.status.set("Edited vectors saved to a new CSV. Existing files were not replaced.")
        except (OSError, ValueError) as error:
            self.status.set(f"Vector save refused: {error}")

    def infer(self, native: bool) -> None:
        """Capture inputs before starting actual retained model/native work."""
        text = self.vectors.get("1.0", "end-1c")
        self.run(lambda: self.workspace.infer(text, include_native=native), self.show_inference)

    def review(self) -> None:
        """Reject invalid/nonfinite thresholds before any scientific work begins."""
        try:
            limits = ReviewLimits(*(float(variable.get()) for variable in self.gates))
        except (TypeError, ValueError):
            self.status.set("Review refused: every gate must be a finite number from 0 to 1.")
            return
        self.run(lambda: self.workspace.review(limits), self.show_review)

    def inspect(self) -> None:
        """Verify current complete artifacts before reporting their identity; no timing
        rerun is implied."""

        def operation():
            verify_bundle(self.workspace.active)
            return {
                "generation": self.workspace.state["active"],
                "status": "Complete artifact fingerprints verified",
                ("notice"): (
                    "Verification does not rerun model quality or certify real-camera performance."
                ),
            }

        self.run(operation, self.show_evidence)

    def rebuild(self) -> None:
        """Keep original deterministic training/export/quality gates and publish a new
        generation only on success."""
        if self.busy or self.closing:
            return
        if not messagebox.askyesno(
            "Rebuild reproducible reference?",
            (
                "This trains the actual original model and exports both ONNX variants in a "
                "new local generation. Prior generations remain. Release thresholds are "
                "unchanged."
            ),
            parent=self.root,
        ):
            return
        self.run(self.workspace.rebuild, self.show_evidence)

    def show_inference(self, payload) -> None:
        """Display per-row actual probabilities and same-reference agreement without
        inventing confidence guarantees."""
        self.predictions.delete(*self.predictions.get_children())
        result = payload["result"]
        for name, runtime in result["runtimes"].items():
            for identifier, asserted, values in zip(
                result["ids"], result["expected_labels"], runtime["probabilities"], strict=True
            ):
                predicted = max(range(3), key=lambda index: values[index])
                self.predictions.insert(
                    "",
                    "end",
                    values=(
                        RUNTIME_NAMES[name],
                        identifier,
                        CLASS_NAMES[predicted],
                        asserted,
                        f"{values[predicted]:.6f}",
                        f"{runtime['reference_max_abs']:.3g}",
                    ),
                )
        self.show_evidence(payload)

    def show_review(self, payload) -> None:
        """Expose weak class/condition failures rather than hiding them behind average accuracy."""
        self.review_rows.delete(*self.review_rows.get_children())
        for name, candidate in payload["result"]["candidates"].items():
            for condition, result in candidate["conditions"].items():
                metrics = result["metrics"]
                worst = min(item["recall"] for item in metrics["per_class"].values())
                self.review_rows.insert(
                    "",
                    "end",
                    values=(
                        RUNTIME_NAMES[name],
                        condition,
                        f"{metrics['accuracy']:.3f}",
                        f"{worst:.3f}",
                        f"{metrics['ece_10_bin']:.4g}",
                        f"{result['reference_max_abs']:.3g}",
                        "pass" if all(result["gates"].values()) else "review failures",
                    ),
                )
        self.show_evidence(payload)

    def show_evidence(self, payload) -> None:
        """Keep full captured evidence for explicit export; truncate only the labelled
        screen preview."""
        self.last_evidence = payload
        text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
        if len(text) > MAX_PREVIEW:
            text = (
                text[:60000]
                + "\n[Screen preview truncated; export retains the full captured result.]"
            )
        self.evidence.configure(state="normal")
        self.evidence.delete("1.0", "end")
        self.evidence.insert("1.0", text)
        self.evidence.configure(state="disabled")

    def export_result(self) -> None:
        """Save exactly the captured result to a new chosen JSON file, never silently overwrite."""
        if self.busy or self.closing or self.last_evidence is None:
            self.status.set("Run or inspect a result before exporting.")
            return
        name = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("Captured evidence", "*.json")], parent=self.root
        )
        if not name:
            return
        try:
            new_json(Path(name), self.last_evidence)
            self.status.set(
                "Captured evidence exported to a new file; model/profile state was unchanged."
            )
        except (OSError, ValueError) as error:
            self.status.set(f"Export refused: {error}")

    def help(self) -> None:
        """Explain real workflow, local privacy and model limitations without network navigation."""
        messagebox.showinfo(
            "Edge Vision · Jason Stys",
            (
                "Prepare/import exact 8x8 vector CSV or draw one synthetic pattern. Run "
                "NumPy/ONNX or all five actual engines, then review every model under four "
                "conditions. Gates are advisory; nothing deploys automatically. "
                "ECE is 10-bin expected calibration error; lower is better. Reference error "
                "compares probabilities with the portable model. Rebuild "
                "runs the original training/export/release checks in a new generation. "
                "Prior generations/reports remain, up to 100 each. Save edited CSV explicitly. "
                "Scroll the input horizontally and predictions vertically. Close normally before "
                "backing up the entire plaintext OS-owner profile. No accounts, network "
                "service, camera or external telemetry are added. Model probabilities and "
                "same-session timings do not certify field performance. Source/contact: "
                "github.com/JasonStys/edge-vision-deployment-benchmark."
            ),
            parent=self.root,
        )

    def run(self, operation: Callable[[], Any], on_result: Callable[[Any], None]) -> None:
        """Retain one real background task and disable edits until its actual completion."""
        if self.busy or self.closing:
            return
        self.busy, self.on_result = True, on_result
        self.status.set("Running actual retained computation; inputs are locked until completion…")
        for control in self.controls:
            control.configure(state="disabled")

        def work():
            try:
                self.results.put((True, operation()))
            except Exception as error:
                self.results.put((False, f"{type(error).__name__}: {error}"))

        self.worker = threading.Thread(target=work, name="edgevision-owned-work", daemon=False)
        self.worker.start()

    def _poll(self) -> None:
        """Join completed work before rendering/unlocking; retain ownership while Quit is
        pending."""
        if self.closed:
            return
        if self.busy and self.worker is not None and not self.worker.is_alive():
            self.worker.join()
            ok, result = self.results.get_nowait()
            self.busy = False
            if ok:
                self.last_evidence = result
                try:
                    if self.on_result is not None:
                        self.on_result(result)
                    self.status.set(
                        "Actual computation completed. Inspect the result and its limitations."
                    )
                except Exception:
                    self.status.set(
                        "Computation completed; display failed. Captured evidence can be exported."
                    )
            else:
                self.status.set("Operation refused; prior active data retained. " + result)
            if not self.closing:
                for control in self.controls:
                    control.configure(state="readonly" if control is self.expected else "normal")
        if self.closing and not self.busy:
            self.workspace.close()
            self.closed = True
            self.root.destroy()
            return
        self.root.after(30, self._poll)

    def close(self) -> None:
        """Keep the window/profile lock until real training/inference/file work actually
        finishes."""
        if self.closed:
            return
        self.closing = True
        for control in self.controls:
            control.configure(state="disabled")
        self.status.set("Closing after real work completes and the local profile lock is released…")
