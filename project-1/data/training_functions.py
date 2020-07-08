def train_basic(vocab_size, max_len, shape, game_settings, seed):
  torch.manual_seed(seed)

  gameBasic_path = "models/basic_game6.pt"

  overwrite = False
  retrain = True
  if retrain:
    senderVision = Vision()
    receiverVision = Vision()

    sender = BasicSender(senderVision, hidden_size)
    receiver = BasicReceiver(receiverVision, hidden_size)

    sender_rnn = RnnSenderGS(sender, vocab_size, emb_size, hidden_size,
                                      cell="gru", max_len=max_len, temperature=.2)
    receiver_rnn = RnnReceiverGS(receiver, vocab_size, emb_size,
                        hidden_size, cell="gru")
    game_rnn = SenderReceiverRnnGS(sender_rnn, receiver_rnn, loss, device)

    optimizer = torch.optim.Adam(game_rnn.parameters(), lr=2e-4)
    with Capturing() as output:
      trainer = Trainer(game=game_rnn, optimizer=optimizer, 
                        train_data=static6_loader,
                        validation_data=test6_loader, 
                        callbacks=[ConsoleLogger(print_train_loss=True, as_json=True)])
      trainer.train(40)
    if not os.path.exists(gameBasic_path) or overwrite:
      torch.save(game_rnn, gameBasic_path)
  else:
    print("Basic game already trained")

  if output is not None:
    training_logs.append(parse_logs(output, game_settings))


def train_att(vocab_size, max_len, shape, game_settings, seed):
  torch.manual_seed(seed)

  gameAtt_path = "models/Att_game6.pt"

  overwrite = False
  retrain = True
  if retrain:
    senderVision = torch.load(visionAtt_path)
    receiverVision = torch.load(visionAtt_path)

    sender = TrainedSender(senderVision, hidden_size)
    receiver = TrainedReceiver(receiverVision, hidden_size)

    sender_rnn = RnnSenderGS(sender, vocab_size, emb_size, hidden_size,
                                      cell="gru", max_len=max_len, temperature=.2)
    receiver_rnn = RnnReceiverGS(receiver, vocab_size, emb_size,
                        hidden_size, cell="gru")
    game_rnn = SenderReceiverRnnGS(sender_rnn, receiver_rnn, loss, device)

    optimizer = torch.optim.Adam(game_rnn.parameters(), lr=2e-4)
    with Capturing() as output:
      trainer = Trainer(game=game_rnn, optimizer=optimizer, 
                        train_data=static6_loader,
                        validation_data=test6_loader, 
                        callbacks=[ConsoleLogger(print_train_loss=True, as_json=True)])
      trainer.train(40)
    if not os.path.exists(gameAtt_path) or overwrite:
      torch.save(game_rnn, gameAtt_path)
  else:
    print("Att game already trained")

  if output is not None:
    training_logs.append(parse_logs(output, game_settings))


def train_base(vocab_size, max_len, shape, game_settings, seed):
  overwrite = False
  retrain = True
  gameBaseline_path = "models/baseline_game6.pt"
  if retrain:
    torch.manual_seed(seed)
    
    senderVision = Vision()
    receiverVision = Vision()

    sender = BasicSender(senderVision, hidden_size)
    receiver = BasicReceiver(receiverVision, hidden_size)

    sender_rnn = RnnSenderGS(sender, vocab_size, emb_size, hidden_size,
                                      cell="gru", max_len=max_len, temperature=0.2)
    receiver_rnn = RnnReceiverGS(receiver, vocab_size, emb_size,
                        hidden_size, cell="gru")
    game_rnn = SenderReceiverRnnGS(sender_rnn, receiver_rnn, loss, device, baseline=True)

    optimizer = torch.optim.Adam(game_rnn.parameters(), lr=2e-4)
    with Capturing() as output:
      trainer = Trainer(game=game_rnn, optimizer=optimizer, 
                        train_data=static6_loader,
                        validation_data=test6_loader, 
                        callbacks=[ConsoleLogger(print_train_loss=True, as_json=True)])
      trainer.train(40)
    if not os.path.exists(gameBaseline_path) or overwrite:
      torch.save(game_rnn, gameBaseline_path)
  else:
    print("Baseline game already trained")

  if output is not None:
    training_logs.append(parse_logs(output, game_settings))