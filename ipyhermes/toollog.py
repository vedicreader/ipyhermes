"Searchable tool-call history using litesearch."
import time
from pathlib import Path

__all__ = ['ToolCallLog']


def _toollog_db(db_path=None):
    "Create or open the tool-call log database."
    from litesearch import database
    if db_path is None:
        from fastcore.xdg import xdg_data_home
        d = xdg_data_home() / 'ipyhermes'
        d.mkdir(parents=True, exist_ok=True)
        db_path = str(d / 'toollog.db')
    return database(db_path, sem_search=False)


class ToolCallLog:
    "Append-only searchable log of tool calls. Uses litesearch FTS5 for full-text search."
    def __init__(self, session_id: str = 'default', db_path: str | None = None):
        self.session_id = session_id
        self._db = _toollog_db(db_path)
        self._store = self._db.get_store('tool_calls')

    def log(self, tool_name: str, input_args: str, output: str,
            duration_ms: float = 0, success: bool = True):
        "Log a tool call. Fire-and-forget."
        import json
        content = f"Tool: {tool_name}\nInput: {input_args}\nOutput: {output}"
        meta = json.dumps(dict(
            tool_name=tool_name, session_id=self.session_id,
            success=success, duration_ms=round(duration_ms, 1),
            ts=time.time(),
        ))
        try:
            self._store.insert(dict(content=content, metadata=meta))
        except Exception:
            pass  # never break the session

    def search(self, query: str, session_id: str | None = None, k: int = 10) -> list[dict]:
        "Full-text search over tool-call history."
        import json
        where, args = None, None
        sid = session_id if session_id is not None else self.session_id
        if sid:
            where = "json_extract(metadata, '$.session_id') = :sid"
            args = dict(sid=sid)
        try:
            sql = self._store.search_sql(order_by='rank', limit=k, where=where)
            rows = list(self._db.execute(sql, dict(query=query, **(args or {}))).fetchall())
            cols = [d[0] for d in self._db.execute(sql, dict(query=query, **(args or {}))).description]
            results = []
            for row in rows:
                rec = dict(zip(cols, row))
                try:
                    rec['metadata'] = json.loads(rec.get('metadata', '{}'))
                except Exception:
                    pass
                results.append(rec)
            return results
        except Exception:
            return []

    def recent(self, n: int = 20, session_id: str | None = None) -> list[dict]:
        "Return the last n tool calls."
        import json
        sid = session_id if session_id is not None else self.session_id
        where_clause = ""
        params = dict(n=n)
        if sid:
            where_clause = "WHERE json_extract(metadata, '$.session_id') = :sid"
            params['sid'] = sid
        try:
            sql = f"SELECT * FROM tool_calls {where_clause} ORDER BY rowid DESC LIMIT :n"
            rows = list(self._db.execute(sql, params).fetchall())
            cols = [d[0] for d in self._db.execute(sql, params).description] if rows else []
            results = []
            for row in rows:
                rec = dict(zip(cols, row))
                try:
                    rec['metadata'] = json.loads(rec.get('metadata', '{}'))
                except Exception:
                    pass
                results.append(rec)
            return results
        except Exception:
            return []
