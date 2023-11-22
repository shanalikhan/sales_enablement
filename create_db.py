import sqlite3

class DatabaseManager:
    def __init__(self, db_name='entities.db'):
        self.db_name = db_name
        self.conn = self.create_connection()

    def create_connection(self):
        conn = sqlite3.connect(self.db_name)
        print('Database connection established successfully')
        return conn

    def close_connection(self):
        self.conn.close()
        print('Database connection closed successfully')

    def get_connection(self):
        return self.conn

    def create_tables(self):
        try:
            cursor = self.conn.cursor()

            # Create a new table for Documents
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS Clients (
                document_id INTEGER PRIMARY KEY,
                document_name TEXT,
                client_name TEXT
            )
            ''')

            print('Table Documents created successfully')

            # Create a new table for Documents
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS Documents (
                document_chunk_id INTEGER PRIMARY KEY,
                document_id INTEGER,
                document_chunk_name TEXT
            )
            ''')
            print('Table Documents created successfully')

            # Create a new table for Entities
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS Entities (
                entity_id INTEGER PRIMARY KEY,
                entity TEXT,
                reference_document TEXT
            )
            ''')
            print('Table Entities created successfully')

            # Commit the changes
            self.conn.commit()
        except sqlite3.OperationalError as e:
            print(f"An error occurred: {e}")

    def insert_into_entities(self, entity, reference_document):
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT INTO Entities (entity, reference_document)
        VALUES (?, ?)
        ''', (entity, reference_document))
        self.conn.commit()

    def insert_into_clients(self, doc_name, client_name):
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT INTO Clients (document_name, client_name)
        VALUES (?, ?)
        ''', (doc_name, client_name))
        self.conn.commit()

    def insert_into_documents(self, doc_id, reference_document):
        cursor = self.conn.cursor()
        cursor.execute('''
        INSERT INTO Documents (document_id, document_chunk_name)
        VALUES (?, ?)
        ''', (doc_id, reference_document))
        self.conn.commit()

    def select_all_from_table(self, table_name='Entities', condition=''):
        cursor = self.conn.cursor()
        cursor.execute(f'SELECT * FROM {table_name} {condition}')
        rows = cursor.fetchall()
        return rows

    def get_entities_from_table(self, metadata, table_name='Entities'):
        cursor = self.conn.cursor()
        cursor.execute(f'SELECT * FROM {table_name} WHERE reference_document = ?', (metadata,))
        rows = cursor.fetchall()
        return rows

# Example usage:
if __name__ == '__main__':
    db_manager = DatabaseManager()
    db_manager.create_tables()

    # Insert some data as an example
    db_manager.insert_into_entities('Entity1', '8_audio (2).txt')
    db_manager.insert_into_documents('Document1', '8_audio (2).txt')

    rows = db_manager.get_entities_from_table('8_audio (2).txt')
    ents = [row[1] for row in rows if row[1] != '']
    print(ents)

    db_manager.close_connection()