
About
=====

For this Python Implementation of the AlphaZero Model of GoogleDeepMind I combined the results of two other repositories(https://github.com/DylanSnyder31/AlphaZero-Chess and https://github.com/Zeta36/chess-alpha-zero)

I used the repository from Zeta36 for the setup of the Self Play, Optimizing and Evaluation Process of the Model. For interaction with a GUI i used the Implementation of DylanSynder31.

You can Genrate new Self-Played Games with 

```bash
python \AlphaZero\run.py --cmd self 
```

You can train that Model on the self played games with 

```bash
python \AlphaZero\run.py --cmd opt 
```

And You can evaluate the new Model with 

```bash
python \AlphaZero\run.py --cmd eval 
```

You will find further instructions in the repository of Zeta36

If you want to Play a Game against the best model so far you can run 

```bash
python \AlphaZero\Play_Agent.py 
```

This will open a GUI on your Computer and you can actually play Chess



