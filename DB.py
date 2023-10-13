from flask import Flask, g
import sqlite3

app = Flask(__name__)

# Configuration for your SQLite database
DATABASE = 'demoSQL.db'  # Change 'your_database.db' to your desired database name
app.config['DATABASE'] = DATABASE

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(app.config['DATABASE'])
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(error):
    if 'db' in g:
        g.db.close()

def init_db():
    db = get_db()
    with app.open_resource('schema.sql', mode='r') as f:
        db.cursor().executescript(f.read())
    db.commit()

# Create the 'movie' table and initialize the database
def create_movie_table():
    db = get_db()
    db.execute('''
        CREATE TABLE IF NOT EXISTS movie (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            link TEXT NOT NULL,
            frames INTEGER NOT NULL,
            slug TEXT NOT NULL
        )
    ''')
    db.commit()

if __name__ == '__main__':
    with app.app_context():
        create_movie_table()
    app.run(debug=True)
