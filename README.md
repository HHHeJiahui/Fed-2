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

First go to the mastodon directory: `cd mastodon/`

Then run Fed-2 with the following command: 
`RAILS_ENV=development Fed2:run_Fed2 trainer=YOUR_TRAINER calculator=YOUR_CALCULATOR gossip_peer_num=YOUR_GOSSIP_PEER_NUM gossip_data_time=YOUR_GOSSIP_DATA_TIME calculate_similarity_time=YOUR_CALCULATE_SIMILARITY_TIME federated_partner_num=YOUR_FEDERATED_PARTNER_NUM federated_learning_time=YOUR_FEDERATED_LEARNING_TIME`

In the command, you need to set some parameters, of course you can use the default value, the following is the description of each parameter:
- **trainer=YOUR_TRAINER**
  - trainer is local model training algorithms
  - Optional values are `NB` (Naïve Bayes) and `LR` (Logistic Regression), the default value is `NB`
  - Example: `trainer=NB`
- **calculator=YOUR_CALCULATOR**
  - calculator is the metric for calculating instance compatibility
  - Optional values are `Hashtags`, `Peers`, `Rules` and `Blocks`, the default value is `Hashtags`
  - Example: `calculator=Hashtags`
- **gossip_peer_num=YOUR_GOSSIP_PEER_NUM**
  - gossip_peer_num is the number of gossip peers per instance
  - If `0 < gossip_peer_num < 1`, it represents the percentage. If `gossip_peer_num > 1`, it represents the actual number. The default value is `0.05`
  - Example: `gossip_peer_num=0.05`, it represents gossip to 5% of peers
- **gossip_data_time=YOUR_GOSSIP_DATA_TIME**
  - gossip_data_time is the time in seconds between each gossip data interval
  - The default value is `3600`
  - Example: `gossip_data_time=3600`
- **calculate_similarity_time=YOUR_CALCULATE_SIMILARITY_TIME**
  - calculate_similarity_time is the time in seconds between each instance of compatibility evaluation
  - The default value is `7200`
  - Example: `calculate_similarity_time=7200`
- **federated_partner_num=YOUR_FEDERATED_PARTNER_NUM**
  - federated_partner_num is the number of federated partners that are performing the FL
  - If `0 < federated_partner_num < 1`, it represents the percentage. If `federated_partner_num > 1`, it represents the actual number. The default value is `0.05`
  - Example: `federated_partner_num=0.05`
- **federated_learning_time=YOUR_FEDERATED_LEARNING_TIME**
  - federated_learning_time is the time in seconds between each FL performed
  - The default value is `10800`
  - Example: `federated_learning_time=10800`
