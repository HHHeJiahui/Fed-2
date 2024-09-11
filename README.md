# Fed-2

Fed-2 is a lightweight gossip-based FL system that does not rely on a central server, allowing Fediverse instances to co-train models with other compatible instances in a privacy-preserving manner. Please follow the steps below to run Fed-2.

## Step 1: Deploy and Run Your Instance

Please refer to the official documentation to deploy an instance (take Mastodon Instance as example), here are some links and documents you can refer to:
- https://github.com/mastodon/mastodon
- https://github.com/McKael/mastodon-documentation/tree/master/Running-Mastodon
- https://github.com/McKael/mastodon-documentation/blob/master/Running-Mastodon/Development-guide.md
- https://github.com/McKael/mastodon-documentation/blob/master/Running-Mastodon/Production-guide.md

You should now be able to run your Mastodon instance with the following instructions:
- Development env: RAILS_ENV=development bundle exec rails server
- Production env: RAILS_ENV=production bundle exec rails server

## Step 2: Install Related Library

`python==3.9.15, Flask==2.2.5, joblib==1.2.0, pandas==1.5.2, APSheduler==3.10.4. scikit-learn==1.2.2, sentence-transformers==2.2.2, numpy==1.23.4, requests==2.28.1, pyarrow==8.0.0`

## Step 3: Setting Instance Information

On lines 20-28 of [app.py](https://github.com/HHHeJiahui/Fed-2/blob/master/Fed-2/app.py), you need to specify:
1. Your Mastodon folder path
2. Your instance name
3. Your pre-labeled training data, use to train local model
4. The embedding model, use to transfer post content to embedding

## Step 4: Move .rake File

Please move [run_Fed2.rake](https://github.com/HHHeJiahui/Fed-2/blob/master/run_Fed2.rake) files to /mastodon/lib/tasks/ path. By doing this you can run Fed-2 through the Mastodon CLI.

## Step 5: Run Fed-2
