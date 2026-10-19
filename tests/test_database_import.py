from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np

from database_import import DatabaseImportError, import_database, read_import_database


def make_database(path, name='Test Person', employee='001', reverse=False):
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("CREATE TABLE people (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, employee_id TEXT NOT NULL DEFAULT '', language TEXT NOT NULL DEFAULT 'zh-CN', welcome_text TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)")
        columns = 'feature_type TEXT, created_at TEXT, feature BLOB, person_id INTEGER, id INTEGER PRIMARY KEY' if reverse else 'id INTEGER PRIMARY KEY, person_id INTEGER, feature BLOB, feature_type TEXT, created_at TEXT'
        connection.execute('CREATE TABLE face_samples (' + columns + ', FOREIGN KEY(person_id) REFERENCES people(id))')
        connection.execute('INSERT INTO people(id,name,employee_id) VALUES (1,?,?)', (name, employee))
        feature = np.ones(512, np.float32).tobytes()
        connection.execute("INSERT INTO face_samples(id,person_id,feature,feature_type,created_at) VALUES (1,1,?,'insightface','2026-10-13')", (feature,))
        connection.commit()


class DatabaseImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / 'source.db'
        self.destination = Path(self.temp.name) / 'target.db'
        make_database(self.source, 'Incoming', '002', reverse=True)
        make_database(self.destination, 'Existing', '001')

    def rows(self, path, table):
        with closing(sqlite3.connect(path)) as connection:
            return connection.execute(f'SELECT * FROM {table} ORDER BY id').fetchall()

    def change_source(self, sql, values=()):
        with closing(sqlite3.connect(self.source)) as connection:
            connection.execute(sql, values)
            connection.commit()

    def test_named_columns_id_remap_and_preimport_backup(self):
        original = self.rows(self.destination, 'people')
        source_bytes = self.source.read_bytes()
        summary = import_database(self.source, self.destination)
        self.assertEqual((summary.people_added, summary.samples_added, summary.people_skipped), (1, 1, 0))
        self.assertEqual(self.rows(summary.backup_path, 'people'), original)
        people = self.rows(self.destination, 'people')
        self.assertEqual(len(people), 2)
        self.assertEqual(people[0], original[0])
        self.assertNotEqual(people[1][0], 1)
        self.assertEqual(self.rows(self.destination, 'face_samples')[1][1], people[1][0])
        self.assertEqual(self.source.read_bytes(), source_bytes)

    def test_repeated_import_skips_conflicting_identity_and_samples(self):
        import_database(self.source, self.destination)
        before = self.rows(self.destination, 'face_samples')
        summary = import_database(self.source, self.destination)
        self.assertEqual((summary.people_added, summary.samples_added, summary.people_skipped), (0, 0, 1))
        self.assertEqual(self.rows(self.destination, 'face_samples'), before)

    def test_same_employee_with_other_name_is_not_merged(self):
        self.change_source("UPDATE people SET employee_id='001'")
        summary = import_database(self.source, self.destination)
        self.assertEqual(summary.people_skipped, 1)
        self.assertEqual(len(self.rows(self.destination, 'face_samples')), 1)

    def test_invalid_samples_are_rejected_before_any_write(self):
        for blob in (b'bad', np.full(512, np.nan, np.float32).tobytes(), np.zeros(512, np.float32).tobytes()):
            with self.subTest(blob_size=len(blob)):
                self.change_source('UPDATE face_samples SET feature=?', (blob,))
                before = self.destination.read_bytes()
                with self.assertRaises(DatabaseImportError):
                    import_database(self.source, self.destination)
                self.assertEqual(self.destination.read_bytes(), before)
        self.assertFalse((self.destination.parent / 'database-backups').exists())

    def test_orphan_samples_are_rejected(self):
        self.change_source('UPDATE face_samples SET person_id=99')
        with self.assertRaises(DatabaseImportError):
            read_import_database(self.source)

    def test_self_import_is_rejected(self):
        with self.assertRaises(DatabaseImportError):
            import_database(self.destination, self.destination)

    def test_insert_failure_rolls_back_people_and_keeps_backup(self):
        with closing(sqlite3.connect(self.destination)) as connection:
            connection.execute("CREATE TRIGGER reject_sample BEFORE INSERT ON face_samples BEGIN SELECT RAISE(ABORT, 'injected failure'); END")
            connection.commit()
        before = self.rows(self.destination, 'people')
        with self.assertRaises(sqlite3.IntegrityError):
            import_database(self.source, self.destination)
        self.assertEqual(self.rows(self.destination, 'people'), before)
        self.assertEqual(len(list((self.destination.parent / 'database-backups').glob('*.db'))), 1)

    def test_wal_backup_contains_committed_personnel(self):
        with closing(sqlite3.connect(self.destination)) as connection:
            connection.execute('PRAGMA journal_mode=WAL')
            connection.execute("INSERT INTO people(name,employee_id) VALUES ('WAL Person','003')")
            connection.commit()
            summary = import_database(self.source, self.destination)
            self.assertEqual(len(self.rows(summary.backup_path, 'people')), 2)


if __name__ == '__main__':
    unittest.main()
