<div id="top"></div>

<br />
<div align="center">
  <h3 align="center">NFL Talent Predictor</h3>
  <p align="center">
    Predict a football player's Madden score using their college and NFL Combine stats.
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
      </ul>
    </li>
    <li>
        <a href="#usage">Usage</a>
        <ul>
            <li><a href="#training">Training</a></li>
            <li><a href="#testing">Testing</a></li>
        </ul>
    </li>
    <li>
        <a href="#future-work">Future Work</a>
    </li>
  </ol>
</details>


## Getting Started

### Prerequisites
Create a Python 3.10 or greater environment and run the following dependencies download command
  ```sh
  pip install -r requirements.txt
  ```

## Usage

### Training

To train a model, run from the src directory
```sh
python train.py <passing|rushing|receiving> [--include_combine]
```

Using the include_combine flag trains on only players who have NFL Combine data.

Additional flags include
```sh
--batch_size (default 4)

--epochs (default 10000)

--lr (default 1e-4)

--train_split (default 0.9)
```
A train all shell script also exists for convenience, and it runs with default values.

### Prediction

To predict a player, run from the src directory
```sh
python predict.py <name> <passing|rushing|receiving> [--include_combine]
```
The name should correspond to an NFL player. Right now the extent of the data only support predictions on players who existed in Madden between 2012 and 2022. In addition, players who were named the same as other players during this time period are not included due to ambiguity (e.g. Adrian Peterson of the Minnesota Vikings and Adrian Peterson of the Chicago Bears).

If the include_combine flag is set, only players who appeared in the NFL Combine will be available.

## Future Work
<ol>
<li>
Clean up and balance the data by using more reliable DBs and other metrics (perhaps ESPN position ratings)
</li>
<li>
Expand to other positions
</li>
<li>
Create in depth errors for when player not found (e.g. duplicate removed, NFL Combine data not found, etc.)
</li>
<li>
See how model performs as classification problem (Classify into rating from 45-99)
</li>
<li>
Implement grid search for hyperparameter tuning
</li>
</ol>

<p align="right">(<a href="#top">back to top</a>)</p>