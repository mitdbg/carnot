# ⚙️ Database Migration Workflow (Alembic)

This guide details the steps required to introduce changes to the SQL database schema (e.g., adding a new column or table) and generate the necessary Alembic migration file.

## Step 1: Modify Your Models

Update your SQLAlchemy models (`.py` files in `app/backend/app/models/`) to include the desired changes (new tables, new columns, etc.).

## Step 2: Generate the Migration File

Run the `generate_migration.sh` script, passing your descriptive message in quotes as the first argument.

**Usage:**

```bash
./generate_migration.sh "descriptive message about the changes"
```
