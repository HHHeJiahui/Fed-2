# Flask_plugin/run_Fed2.rake
namespace :Fed2 do
  desc "Load instance data"
  task run_Fed2: :environment do
    require 'net/http'

    trainer = ENV['trainer'] || 'NB'
    calculator = ENV['calculator'] || 'Hashtags'
    gossip_peer_num = ENV['gossip_peer_num'] || '0.05'
    gossip_data_time = ENV['gossip_data_time'] || '3600'
    calculate_similarity_time = ENV['calculate_similarity_time'] || '7200'
    federated_partner_num = ENV['federated_partner_num'] || '0.05'
    federated_learning_time = ENV['federated_learning_time'] || '10800'

    params = [trainer, calculator, gossip_peer_num, gossip_data_time, calculate_similarity_time, federated_partner_num, federated_learning_time]

    current_dir = File.expand_path(File.dirname(__FILE__))
    python_command = "python3 #{current_dir}/../../Fed-2/Fed-2/app.py " + params.join(' ')
    sh python_command
  end
end