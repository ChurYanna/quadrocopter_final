from __future__ import annotations

import queue
import subprocess
import threading
from typing import Any


class SemanticTracePanel:
    """Small live Tk panel for showing the LLM semantic evidence chain.

    The panel is deliberately optional.  If Tk or a display is unavailable,
    `start()` simply returns False and the recorder continues writing JSON logs.
    """

    def __init__(self, title: str = 'SV-STCP LLM Semantic Passage Monitor'):
        self.title = str(title)
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._started = threading.Event()
        self._closed = threading.Event()
        self._failed = False

    def start(self) -> bool:
        if self._thread is not None:
            return not self._failed
        self._thread = threading.Thread(target=self._run, name='semantic-trace-panel', daemon=True)
        self._thread.start()
        self._started.wait(timeout=2.0)
        return not self._failed

    def submit(self, event: dict[str, Any]) -> None:
        if self._thread is None and not self.start():
            return
        if self._failed:
            return
        self._queue.put(event)

    def close(self) -> None:
        if self._thread is not None and not self._failed:
            self._queue.put(None)

    def wait_until_closed(self) -> None:
        if self._thread is None or self._failed:
            return
        self._closed.wait()

    def _run(self) -> None:
        try:
            import tkinter as tk
            from tkinter import ttk
            import tkinter.font as tkfont
        except Exception:
            self._failed = True
            self._started.set()
            return

        try:
            root = tk.Tk()
            root.title(self.title)
            root.geometry('1180x760+40+40')
            root.minsize(980, 660)
            self._font_family = self._choose_font(root)
            self._install_chinese_font_defaults(root, tkfont, self._font_family)
            print(f'[SEMANTIC_PANEL] using font: {self._font_family}')

            style = ttk.Style(root)
            try:
                style.theme_use('clam')
            except tk.TclError:
                pass
            style.configure('Title.TLabel', font=(self._font_family, 16, 'bold'))
            style.configure('Status.TLabel', font=(self._font_family, 10))
            style.configure('Panel.TLabelframe.Label', font=(self._font_family, 10, 'bold'))

            root.columnconfigure(0, weight=1)
            root.rowconfigure(1, weight=1)

            header = ttk.Frame(root, padding=(12, 10, 12, 6))
            header.grid(row=0, column=0, sticky='ew')
            header.columnconfigure(0, weight=1)
            title = ttk.Label(header, text='SV-STCP 语义证据链实时面板', style='Title.TLabel')
            title.grid(row=0, column=0, sticky='w')
            self._status_var = tk.StringVar(value='等待语义事件...')
            status = ttk.Label(header, textvariable=self._status_var, style='Status.TLabel')
            status.grid(row=1, column=0, sticky='w', pady=(4, 0))

            main = ttk.Frame(root, padding=(12, 4, 12, 12))
            main.grid(row=1, column=0, sticky='nsew')
            main.columnconfigure(0, weight=1)
            main.columnconfigure(1, weight=1)
            main.rowconfigure(0, weight=1)
            main.rowconfigure(1, weight=1)

            self._scene_text = self._make_text_frame(
                main,
                '1. 场景语义 / LLM 输入',
                row=0,
                column=0,
                accent='#174ea6',
            )
            self._strategy_text = self._make_text_frame(
                main,
                '2. LLM 宏观策略 / 安全验证',
                row=0,
                column=1,
                accent='#0b8043',
            )
            self._execution_text = self._make_text_frame(
                main,
                '3. 底层执行反馈',
                row=1,
                column=0,
                accent='#a14200',
            )
            self._timeline_text = self._make_text_frame(
                main,
                '4. 事件时间线',
                row=1,
                column=1,
                accent='#5f2385',
            )

            self._scene_buffer: list[str] = []
            self._strategy_buffer: list[str] = []
            self._execution_buffer: list[str] = []
            self._timeline_buffer: list[str] = []
            self._event_count = 0

            self._started.set()
            root.after(120, lambda: self._poll(root))
            root.mainloop()
        except Exception:
            self._failed = True
            self._started.set()
        finally:
            self._closed.set()

    def _make_text_frame(self, parent: Any, title: str, row: int, column: int, accent: str):
        import tkinter as tk
        from tkinter import ttk

        frame = ttk.LabelFrame(parent, text=title, padding=(8, 6, 8, 8), style='Panel.TLabelframe')
        frame.grid(row=row, column=column, sticky='nsew', padx=6, pady=6)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        bar = tk.Frame(frame, height=4, background=accent)
        bar.grid(row=0, column=0, sticky='ew', pady=(0, 6))

        text = tk.Text(
            frame,
            wrap='word',
            height=13,
            font=(self._font_family, 10),
            relief='flat',
            background='#fbfbfb',
            foreground='#202124',
            insertwidth=0,
            padx=9,
            pady=8,
            spacing1=2,
            spacing2=1,
            spacing3=4,
        )
        text.grid(row=1, column=0, sticky='nsew')
        scrollbar = ttk.Scrollbar(frame, orient='vertical', command=text.yview)
        scrollbar.grid(row=1, column=1, sticky='ns')
        text.configure(yscrollcommand=scrollbar.set)
        text.configure(state='disabled')
        return text

    def _poll(self, root: Any) -> None:
        while True:
            try:
                event = self._queue.get_nowait()
            except queue.Empty:
                break
            if event is None:
                root.destroy()
                return
            self._handle_event(event)
        root.after(120, lambda: self._poll(root))

    def _handle_event(self, event: dict[str, Any]) -> None:
        self._event_count += 1
        title = str(event.get('title', '语义事件'))
        category = str(event.get('category', 'unknown'))
        summary = str(event.get('summary', ''))
        sim_time = event.get('sim_time')
        time_text = '初始化' if sim_time is None else f't={float(sim_time):.2f}s'
        details = event.get('details', {})

        block = self._format_block(title, time_text, summary, details)
        timeline_line = f'[{self._event_count:02d}] {time_text} | {title}: {summary}'
        self._append(self._timeline_text, self._timeline_buffer, timeline_line, max_blocks=40)

        if category in {'scene_semantics', 'llm_input', 'runtime_semantics', 'online_perception'}:
            self._append(self._scene_text, self._scene_buffer, block, max_blocks=8)
        elif category in {'deterministic_plan', 'llm_strategy_output', 'strategy_plan', 'safety_validation', 'online_replanning'}:
            self._append(self._strategy_text, self._strategy_buffer, block, max_blocks=8)
        else:
            self._append(self._execution_text, self._execution_buffer, block, max_blocks=16)
        self._status_var.set(f'已接收 {self._event_count} 个语义事件 | 最新：{title} | {time_text}')

    def _format_block(
        self,
        title: str,
        time_text: str,
        summary: str,
        details: dict[str, Any],
    ) -> str:
        lines = [f'{title}  ({time_text})', summary]
        for key, value in details.items():
            if isinstance(value, list):
                lines.append(f'- {key}:')
                for item in value[:12]:
                    lines.append(f'  - {item}')
                if len(value) > 12:
                    lines.append(f'  - ... 还有 {len(value) - 12} 项')
            else:
                lines.append(f'- {key}: {value}')
        return '\n'.join(lines)

    def _append(self, widget: Any, buffer: list[str], block: str, max_blocks: int) -> None:
        buffer.append(block)
        del buffer[:-max_blocks]
        widget.configure(state='normal')
        widget.delete('1.0', 'end')
        widget.insert('end', '\n\n'.join(buffer))
        widget.see('end')
        widget.configure(state='disabled')

    @staticmethod
    def _choose_font(root: Any) -> str:
        preferred = (
            'Noto Sans CJK SC',
            'Noto Sans CJK TC',
            'Noto Sans CJK JP',
            'WenQuanYi Micro Hei',
            'WenQuanYi Zen Hei',
            'Source Han Sans SC',
            'Microsoft YaHei',
            'SimHei',
        )
        try:
            output = subprocess.check_output(
                ['fc-match', '-f', '%{family}', 'Noto Sans CJK SC'],
                text=True,
                timeout=1.0,
            )
            family = output.split(',')[0].strip()
            if family:
                return family
        except Exception:
            pass
        try:
            import tkinter.font as tkfont

            families = set(tkfont.families(root))
            for name in preferred:
                if name in families:
                    return name
        except Exception:
            families = set()
        try:
            output = subprocess.check_output(
                ['fc-match', '-f', '%{family}', ':lang=zh'],
                text=True,
                timeout=1.0,
            )
            for family in output.split(','):
                family = family.strip()
                if family and (not families or family in families):
                    return family
        except Exception:
            pass
        return 'TkDefaultFont'

    @staticmethod
    def _install_chinese_font_defaults(root: Any, tkfont: Any, family: str) -> None:
        """Force Tk named fonts to a CJK-capable family.

        Tk widgets can silently fall back to a Latin-only default even when a
        Text widget is configured with another font.  Updating named fonts makes
        labels, labelframe titles, scrollbars, and text content use the same
        Chinese-capable family.
        """
        if not family or family == 'TkDefaultFont':
            return
        for name, size in (
            ('TkDefaultFont', 10),
            ('TkTextFont', 10),
            ('TkMenuFont', 10),
            ('TkHeadingFont', 10),
            ('TkCaptionFont', 10),
            ('TkSmallCaptionFont', 9),
            ('TkIconFont', 10),
            ('TkTooltipFont', 9),
        ):
            try:
                tkfont.nametofont(name).configure(family=family, size=size)
            except Exception:
                pass
        try:
            root.option_add('*Font', (family, 10))
        except Exception:
            pass
