from __future__ import annotations

import queue
import subprocess
import threading
from typing import Any


class SemanticTracePanel:
    """Small live Tk panel for showing the SFSC + VLM evidence chain.

    The panel is deliberately optional.  If Tk or a display is unavailable,
    `start()` simply returns False and the recorder continues writing JSON logs.
    """

    def __init__(self, title: str = 'SV-STCP VLM Semantic Passage Monitor'):
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
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

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
            title = ttk.Label(header, text='SV-STCP SFSC + VLM 多模态策略证据链实时面板', style='Title.TLabel')
            title.grid(row=0, column=0, sticky='w')
            self._status_var = tk.StringVar(value='等待 SFSC/VLM 事件...')
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
                '1. 兜底策略 / 当前执行',
                row=0,
                column=0,
                accent='#174ea6',
            )
            self._strategy_text = self._make_text_frame(
                main,
                '2. VLM 是否起作用',
                row=0,
                column=1,
                accent='#0b8043',
            )
            self._execution_text = self._make_text_frame(
                main,
                '3. 实际进入底层的改动',
                row=1,
                column=0,
                accent='#a14200',
            )
            self._timeline_text = self._make_text_frame(
                main,
                '4. 详细记录 / 证据链',
                row=1,
                column=1,
                accent='#5f2385',
            )

            self._scene_buffer: list[str] = []
            self._strategy_buffer: list[str] = []
            self._execution_buffer: list[str] = []
            self._timeline_buffer: list[str] = []
            self._uav_status: dict[str, str] = {}
            self._fallback_status: dict[str, str] = {}
            self._event_count = 0
            self._render_overview()

            self._started.set()
            root.after(120, lambda: self._poll(root))
            root.mainloop()
        except Exception:
            self._failed = True
            self._started.set()
        finally:
            for attr in (
                '_status_var',
                '_scene_text',
                '_strategy_text',
                '_execution_text',
                '_timeline_text',
                '_scene_buffer',
                '_strategy_buffer',
                '_execution_buffer',
                '_timeline_buffer',
                '_uav_status',
                '_fallback_status',
            ):
                try:
                    setattr(self, attr, None)
                except Exception:
                    pass
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
        title = str(event.get('title', 'SFSC/VLM 事件'))
        category = str(event.get('category', 'unknown'))
        summary = str(event.get('summary', ''))
        sim_time = event.get('sim_time')
        time_text = '初始化' if sim_time is None else f't={float(sim_time):.2f}s'
        details = event.get('details', {})

        block = self._format_block(category, title, time_text, summary, details)
        timeline_line = f'[{self._event_count:02d}] {time_text} | {title}: {summary}'
        self._append(self._timeline_text, self._timeline_buffer, timeline_line, max_blocks=40)

        if category == 'execution_feedback' and title == '单机执行模式切换':
            self._update_uav_status(details, time_text)
        elif category == 'online_replanning':
            self._update_fallback_status(details)
            self._append(
                self._strategy_text,
                self._strategy_buffer,
                self._format_vlm_candidate_summary(title, time_text, details),
                max_blocks=8,
            )
        elif category in {
            'scene_semantics',
            'llm_input',
            'runtime_semantics',
            'online_perception',
            'visual_observation',
            'multimodal_vlm_input',
        }:
            self._append(self._timeline_text, self._timeline_buffer, block, max_blocks=40)
        elif category in {
            'deterministic_plan',
            'llm_strategy_output',
            'vlm_strategy_output',
            'strategy_plan',
            'safety_validation',
        }:
            self._append(self._strategy_text, self._strategy_buffer, block, max_blocks=12)
        elif category == 'execution_feedback' and title.startswith('VLM 策略'):
            self._append(
                self._execution_text,
                self._execution_buffer,
                self._format_execution_admission_summary(title, time_text, details),
                max_blocks=8,
            )
        else:
            self._append(self._execution_text, self._execution_buffer, block, max_blocks=10)
        self._status_var.set(f'已接收 {self._event_count} 个 SFSC/VLM 事件 | 最新：{title} | {time_text}')

    def _update_uav_status(self, details: dict[str, Any], time_text: str) -> None:
        uav = str(details.get('无人机', 'UAV ?'))
        mode = str(details.get('执行模式', '未知模式'))
        target = str(details.get('当前目标', '未知目标'))
        route = details.get('绕行方式') or details.get('目标策略') or details.get('含义')
        center_x = details.get('障碍中心 x')
        extra_parts = []
        if route is not None:
            extra_parts.append(f'策略={route}')
        if center_x is not None:
            extra_parts.append(f'障碍x={center_x}')
        extra_text = f' | {" | ".join(extra_parts)}' if extra_parts else ''
        self._uav_status[uav] = f'{uav}: {mode} -> {target}{extra_text}  ({time_text})'
        self._render_overview()

    def _update_fallback_status(self, details: dict[str, Any]) -> None:
        for item in details.get('确定性兜底策略', []) if isinstance(details.get('确定性兜底策略'), list) else []:
            obstacle_id = str(item).split(':', 1)[0].strip()
            if obstacle_id:
                self._fallback_status[obstacle_id] = str(item)
        self._render_overview()

    def _render_overview(self) -> None:
        lines = ['确定性兜底策略（当前/近期障碍）:']
        if self._fallback_status:
            for obstacle_id in sorted(self._fallback_status):
                lines.append(f'- {self._fallback_status[obstacle_id]}')
        else:
            lines.append('- 等待在线兜底策略事件...')

        lines.append('')
        lines.append('当前无人机执行态:')
        if self._uav_status:
            for key in sorted(self._uav_status, key=self._uav_sort_key):
                lines.append(f'- {self._uav_status[key]}')
        else:
            lines.append('- 等待底层执行模式切换...')
        self._append(self._scene_text, self._scene_buffer, '\n'.join(lines), max_blocks=1)

    @staticmethod
    def _format_vlm_candidate_summary(title: str, time_text: str, details: dict[str, Any]) -> str:
        obstacles = ', '.join(str(item) for item in details.get('障碍集合', [])) or 'unknown'
        changed = details.get('VLM是否改变兜底策略', '未知')
        fallback = details.get('确定性兜底策略', [])
        changes = details.get('策略变化摘要', [])
        policy_changes = [item for item in changes if 'policy' in str(item)]
        window_changes = [item for item in changes if item not in policy_changes]
        lines = [
            f'{title}  ({time_text})',
            f'覆盖障碍: {obstacles}',
            f'VLM是否起作用: {changed}',
            '原本兜底:',
        ]
        lines.extend(f'- {item}' for item in (fallback[:4] if isinstance(fallback, list) else [fallback]))
        lines.append('核心策略改变:')
        if policy_changes:
            lines.extend(f'- {item}' for item in policy_changes[:5])
        else:
            lines.append('- 无策略族改变')
        if window_changes:
            lines.append('窗口/距离微调:')
            lines.extend(f'- {item}' for item in window_changes[:4])
        admission = details.get('执行准入状态')
        if admission:
            lines.append(f'准入状态: {admission}')
        return '\n'.join(lines)

    @staticmethod
    def _format_execution_admission_summary(title: str, time_text: str, details: dict[str, Any]) -> str:
        raw = details.get('VLM原始建议', [])
        actual = details.get('实际进入底层执行', [])
        runtime = details.get('实体绕障运行时同步', [])
        validation = details.get('安全验证', '未知')
        timeliness = details.get('时效性验证', [])
        ttc_lines = [
            item for item in timeliness
            if isinstance(item, str) and ('TTC=' in item or '时效性通过' in item or '不足' in item)
        ]
        lines = [
            f'{title}  ({time_text})',
            f'安全验证: {validation}',
            'VLM原始建议:',
        ]
        lines.extend(f'- {item}' for item in (raw[:5] if isinstance(raw, list) else [raw]))
        lines.append('实际进入底层执行:')
        lines.extend(f'- {item}' for item in (actual[:6] if isinstance(actual, list) else [actual]))
        if runtime:
            lines.append('实体绕障运行时同步:')
            lines.extend(f'- {item}' for item in (runtime[:5] if isinstance(runtime, list) else [runtime]))
        if ttc_lines:
            lines.append('TTC/时效性:')
            lines.extend(f'- {item}' for item in ttc_lines[:4])
        return '\n'.join(lines)

    @staticmethod
    def _uav_sort_key(label: str) -> tuple[int, str]:
        digits = ''.join(ch for ch in str(label) if ch.isdigit())
        return (int(digits) if digits else 999, str(label))

    def _format_block(
        self,
        category: str,
        title: str,
        time_text: str,
        summary: str,
        details: dict[str, Any],
    ) -> str:
        lines = [f'{title}  ({time_text})', summary]
        for key, value in self._panel_details(category, details).items():
            if isinstance(value, list):
                lines.append(f'- {key}:')
                for item in value[:6]:
                    lines.append(f'  - {item}')
                if len(value) > 6:
                    lines.append(f'  - ... 还有 {len(value) - 6} 项，完整内容见日志')
            else:
                lines.append(f'- {key}: {value}')
        return '\n'.join(lines)

    @staticmethod
    def _panel_details(category: str, details: dict[str, Any]) -> dict[str, Any]:
        key_order_by_category = {
            'online_replanning': (
                '链路阶段',
                '一句话说明',
                '事件类型',
                '障碍集合',
                '外部模型耗时',
                '前视图路径',
                '视觉来源',
                '相机名称',
                'VLM是否改变兜底策略',
                '策略变化摘要',
                '执行准入状态',
                '验证通过',
                '是否使用回退',
                '当前缓存覆盖',
            ),
            'execution_feedback': (
                '算法链路',
                'VLM候选计划编号',
                'VLM候选来源',
                '实际执行计划编号',
                '实际执行来源',
                'VLM原始建议',
                '安全验证',
                '时效性验证',
                '安全投影',
                '实际进入底层执行',
                '实体绕障运行时同步',
                '当前通行顺序',
                '作用方式',
                '执行阶段',
                '任务是否成功',
                '总耗时',
                '最小机间距',
                '最小洞口净空',
            ),
            'multimodal_vlm_input': (
                '链路阶段',
                '一句话说明',
                '前视图路径',
                '视觉来源',
                '相机名称',
                '融合模态',
                '引用SFSC事件',
                '输入包路径',
            ),
            'vlm_strategy_output': (
                '链路阶段',
                '一句话说明',
                '来源',
                '置信度',
                '建议模式',
                '目标策略',
                'SFSC一致性',
                '遮挡风险',
                '形态风险',
                '是否进入控制',
                '结果文件路径',
            ),
        }
        key_order = key_order_by_category.get(str(category))
        if key_order is None:
            return details
        compact = {key: details[key] for key in key_order if key in details}
        return compact or details

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
