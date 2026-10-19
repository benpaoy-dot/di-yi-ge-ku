"""Read-only validation and transactional, non-overwriting personnel imports."""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sqlite3
import uuid

import numpy as np


class DatabaseImportError(ValueError):
    pass


@dataclass(frozen=True)
class ImportSummary:
    people_added: int
    samples_added: int
    people_skipped: int
    backup_path: str


@dataclass(frozen=True)
class ImportData:
    people: tuple
    samples: tuple


PEOPLE_COLUMNS = ('id', 'name', 'employee_id', 'language', 'welcome_text', 'created_at')
SAMPLE_COLUMNS = ('id', 'person_id', 'feature', 'feature_type', 'created_at')
FEATURE_DIMENSIONS = {'insightface': 512, 'sface': 128, 'legacy': 2304}


def read_import_database(path: str | Path) -> ImportData:
    source_path = Path(path).resolve(strict=True)
    if not source_path.is_file():
        raise DatabaseImportError('请选择 SQLite 数据库文件')
    uri = source_path.as_uri() + '?mode=ro'
    try:
        with closing(sqlite3.connect(uri, uri=True, timeout=5)) as source, closing(sqlite3.connect(':memory:')) as snapshot:
            source.backup(snapshot)
            if snapshot.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise DatabaseImportError('来源数据库完整性检查未通过')
            for table, required in (('people', PEOPLE_COLUMNS), ('face_samples', SAMPLE_COLUMNS)):
                columns = {row[1] for row in snapshot.execute(f'PRAGMA table_info({table})')}
                if not set(required).issubset(columns):
                    raise DatabaseImportError(f'来源数据库缺少 {table} 的必要字段，不支持此格式')
            people = tuple(snapshot.execute('SELECT ' + ','.join(PEOPLE_COLUMNS) + ' FROM people ORDER BY id'))
            samples = tuple(snapshot.execute('SELECT ' + ','.join(SAMPLE_COLUMNS) + ' FROM face_samples ORDER BY id'))
    except sqlite3.Error as error:
        raise DatabaseImportError('无法读取有效的项目 SQLite 数据库') from error
    ids = set()
    names = set()
    employees = set()
    for person_id, name, employee_id, language, welcome_text, created_at in people:
        if not isinstance(person_id, int) or person_id in ids:
            raise DatabaseImportError('人员编号重复或无效')
        if not isinstance(name, str) or not name.strip() or name in names:
            raise DatabaseImportError('来源姓名为空或重复')
        if not all(isinstance(value, str) for value in (employee_id, language, welcome_text, created_at)):
            raise DatabaseImportError('人员字段类型无效')
        if language not in ('zh-CN', 'en-US', 'zh-en'):
            raise DatabaseImportError('来源包含不支持的欢迎语语言')
        if employee_id and employee_id in employees:
            raise DatabaseImportError('来源包含重复员工编号，请先核对身份')
        ids.add(person_id)
        names.add(name)
        if employee_id:
            employees.add(employee_id)
    sample_ids = set()
    for sample_id, person_id, blob, feature_type, created_at in samples:
        if not isinstance(sample_id, int) or sample_id in sample_ids or person_id not in ids:
            raise DatabaseImportError('来源样本编号无效或存在孤立样本')
        sample_ids.add(sample_id)
        if not isinstance(feature_type, str) or feature_type not in FEATURE_DIMENSIONS:
            raise DatabaseImportError('来源包含未知特征类型；不能自动转换识别模型')
        dimension = FEATURE_DIMENSIONS[feature_type]
        if not isinstance(blob, bytes) or len(blob) != dimension * 4:
            raise DatabaseImportError(f'{feature_type} 样本维度无效')
        feature = np.frombuffer(blob, dtype=np.float32)
        if not np.all(np.isfinite(feature)) or not 0 < float(np.linalg.norm(feature)) < float('inf'):
            raise DatabaseImportError('来源包含非有限值或零向量样本')
        if not isinstance(created_at, str):
            raise DatabaseImportError('样本日期字段无效')
    return ImportData(people, samples)


def import_database(source_path: str | Path, destination_path: str | Path) -> ImportSummary:
    source = Path(source_path).resolve(strict=True)
    destination = Path(destination_path).resolve(strict=True)
    if source.samefile(destination):
        raise DatabaseImportError('不能把当前人员数据库导入自身')
    data = read_import_database(source)
    backup_dir = destination.parent / 'database-backups'
    backup_dir.mkdir(exist_ok=True)
    backup = backup_dir / f'face_welcome-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}.db'
    with closing(sqlite3.connect(destination, timeout=5)) as target:
        target.execute('PRAGMA foreign_keys=ON')
        target.execute('BEGIN IMMEDIATE')
        try:
            # Separate reader sees the pre-import state, including committed WAL data.
            with closing(sqlite3.connect(destination, timeout=5)) as reader, closing(sqlite3.connect(backup)) as saved:
                reader.backup(saved)
            existing = target.execute('SELECT name,employee_id FROM people').fetchall()
            names = {row[0] for row in existing}
            employees = {row[1] for row in existing if row[1]}
            id_map = {}
            skipped = 0
            for old_id, name, employee_id, language, welcome_text, created_at in data.people:
                # Never attach faces to an existing name: names alone do not establish identity.
                if name in names or (employee_id and employee_id in employees):
                    skipped += 1
                    continue
                cursor = target.execute(
                    'INSERT INTO people(name,employee_id,language,welcome_text,created_at) VALUES (?,?,?,?,?)',
                    (name, employee_id, language, welcome_text, created_at),
                )
                id_map[old_id] = cursor.lastrowid
                names.add(name)
                if employee_id:
                    employees.add(employee_id)
            added_samples = 0
            for _, person_id, feature, feature_type, created_at in data.samples:
                if person_id not in id_map:
                    continue
                target.execute(
                    'INSERT INTO face_samples(person_id,feature,feature_type,created_at) VALUES (?,?,?,?)',
                    (id_map[person_id], feature, feature_type, created_at),
                )
                added_samples += 1
            if target.execute('PRAGMA foreign_key_check').fetchone() is not None:
                raise DatabaseImportError('导入后关联检查未通过，已回滚')
            target.commit()
        except Exception:
            target.rollback()
            raise
    return ImportSummary(len(id_map), added_samples, skipped, str(backup))
