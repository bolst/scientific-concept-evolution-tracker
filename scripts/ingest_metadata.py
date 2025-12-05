import os, json
from datetime import datetime
from tqdm import tqdm

import _src
from scet.core.db import SessionLocal, engine
from scet.core.models import Paper, Base

from dotenv import load_dotenv
load_dotenv()


LIMIT = 1000
DATA_PATH = os.path.join(os.path.dirname(__file__), '../notebooks/data/arxiv-metadata-oai-snapshot.json')



def parse_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except:
        return None

def get_v1_date(versions):
    if not versions:
        return None
    for v in versions:
        if v['version'] == 'v1':
            # e.g. format: "Mon, 2 Apr 2007 19:18:42 GMT"
            try:
                return datetime.strptime(v['created'], "%a, %d %b %Y %H:%M:%S %Z").date()
            except:
                return None
    return None

def ingest_metadata(data_path=DATA_PATH, limit=LIMIT):
    print("Starting ingestion with " + (f"limit: {limit}" if limit else "no limit"))
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    db = SessionLocal()
    
    # ensure tables exist
    Base.metadata.create_all(bind=engine)

    # don't load everything: stream line-by-line
    count = 0
    
    # want to sample for a diverse set of years...
    # Strategy: skip N records between each ingestion to sample across the file (assuming file is sorted by year)
    # The file is roughly 2.4M lines. If limit is 1000, we want to sample every ~2400th line

    ESTIMATED_TOTAL = 2_400_000
    step_size = max(1, ESTIMATED_TOTAL // limit) if limit else 1
    
    print(f"Taking 1 paper every {step_size} lines.")

    # Initialize progress bar
    total_to_process = limit if limit else ESTIMATED_TOTAL
    pbar = tqdm(total=total_to_process, desc="Ingesting Metadata", unit="paper")

    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if limit and count >= limit:
                break
            
            # Only process if it matches our step
            if i % step_size != 0:
                continue
            
            entry = json.loads(line)
            
            # extract fields
            arxiv_id = entry.get('id')
            title = entry.get('title', '').strip()
            abstract = entry.get('abstract', '').strip()
            categories = entry.get('categories', '').split(' ')
            primary_category = categories[0] if categories else None
            doi = entry.get('doi')
            license_url = entry.get('license')
            update_date_str = entry.get('update_date')
            versions = entry.get('versions', [])
            
            published_date = get_v1_date(versions)
            last_updated_date = parse_date(update_date_str)
            
            # build model from extracted fields
            paper = Paper(
                arxiv_id=arxiv_id,
                title=title,
                abstract=abstract,
                primary_category=primary_category,
                published_date=published_date,
                last_updated_date=last_updated_date,
                doi=doi,
                license=license_url,
                processed_status='pending'
            )
            
            # list of lists: [["Surname", "Firstname", "Suffix"], ...]
            authors_parsed = entry.get('authors_parsed', [])
            for author_entry in authors_parsed:
                parts = [p for p in author_entry if p]
                # from what I understand: [Surname, Firstname, Suffix] -> Firstname Surname
                if len(parts) >= 2:
                    name = f"{parts[1]} {parts[0]}"
                else:
                    name = parts[0]
                # TODO: handle authors (not sure if we even need to?)
                pass

            try:
                # use merge to handle duplicates
                db.merge(paper)
                count += 1
                if count % 100 == 0:
                    db.commit()
                    pbar.update(100)
            except Exception as e:
                print(f"Error processing {arxiv_id}: {e}")
                db.rollback()

    # Update remaining count
    pbar.update(count % 100)
    pbar.close()
    db.commit()
    db.close()
    print(f"Finished processing total of {count} papers")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest ArXiv metadata into the database.")
    parser.add_argument("--data-path", type=str, default=DATA_PATH, help="Path to the arxiv metadata json file")
    parser.add_argument("--limit", type=int, default=None, help="Limit of metadata to process. Will attempt to sample the amount of data over all available years.")
    args = parser.parse_args()
    
    ingest_metadata(args.data_path, args.limit)
