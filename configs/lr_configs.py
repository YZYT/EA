SGD_config = {
  'optimizer':{
    "lr": 0.1, 
    "momentum": 0.9,
    "nesterov": True,
    'weight_decay': 5e-4
  },
  'scheduler':{
      "milestones": [60, 120, 160],
      "gamma": 0.2
  }

}

LA_config = {
  "la_steps": 5, 
  "la_alpha": 0.8,
  "pullback_momentum": "pullback"
}

SWA_config = {
  "SWA_lr": 0.05,
  "SWA_ratio":0.75
}
