# Data Preparation

This is code to split `data-labeled.csv` into training and validation sets


```python
import pandas as pd
import numpy as np
```


```python
df = pd.read_csv('./data-labeled.csv')
```

Creating new column in df and labeling first 800 rows as valid set and remaining as train


```python
df.loc[:800,'Set'] = "valid"
df.loc[800:,'Set'] = 'train'
```


```python
df.head(804)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label</th>
      <th>Text</th>
      <th>Set</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>negative</td>
      <td>That’s the last thing I want to be seen as.</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>1</th>
      <td>positive</td>
      <td>But today, I learned that not everyone just wa...</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>2</th>
      <td>neutral</td>
      <td>Now, the hole’s that much smaller.</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>3</th>
      <td>positive</td>
      <td>I’ve had a couple shots with friends in differ...</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>4</th>
      <td>neutral</td>
      <td>He was grateful for his wife's submission and ...</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>5</th>
      <td>negative</td>
      <td>Somehow, someone else in my life always ends u...</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>6</th>
      <td>positive</td>
      <td>I'm so very happy to say the Ghost has officia...</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>7</th>
      <td>negative</td>
      <td>It’s all SO fucking hard.</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>8</th>
      <td>negative</td>
      <td>I sit alone in my room watching tv and drinkin...</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>9</th>
      <td>positive</td>
      <td>At the end I was glad it was over.</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>10</th>
      <td>negative</td>
      <td>All I wanted to do was curl up in a ball in be...</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>11</th>
      <td>negative</td>
      <td>These emotional rollercoasters of days are sta...</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>12</th>
      <td>negative</td>
      <td>My mom thought I was overreacting.</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>13</th>
      <td>negative</td>
      <td>I really want R to like me.</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>14</th>
      <td>neutral</td>
      <td>After a while he mentioned doing a trip.</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>15</th>
      <td>neutral</td>
      <td>I’d even consider us friends, but I don’t thin...</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>16</th>
      <td>neutral</td>
      <td>And not only lost it, but to the extreme.</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>17</th>
      <td>neutral</td>
      <td>Needed sleep in the afternoon fell asleep a li...</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>18</th>
      <td>negative</td>
      <td>She’s made a few very dry suicide jokes and I’...</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>19</th>
      <td>neutral</td>
      <td>And that was the end of that.</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>20</th>
      <td>neutral</td>
      <td>Alright, that's all for now.</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>21</th>
      <td>neutral</td>
      <td>Leaving dad again and going back to work and r...</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>22</th>
      <td>negative</td>
      <td>im lonely and horny.</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>23</th>
      <td>positive</td>
      <td>She is the best.</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>24</th>
      <td>positive</td>
      <td>c. HE’s an amazing artist, person, teacher, ev...</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>25</th>
      <td>neutral</td>
      <td>Maybe the first time I didn't put it away prop...</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>26</th>
      <td>positive</td>
      <td>I listened to music and was quite at peace bei...</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>27</th>
      <td>neutral</td>
      <td>I don't see why not says for me to put the app...</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>28</th>
      <td>neutral</td>
      <td>Every time!</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>29</th>
      <td>neutral</td>
      <td>(Unless they needed something.)</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>774</th>
      <td>negative</td>
      <td>So fucking fast.</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>775</th>
      <td>neutral</td>
      <td>It’s not even like I wanna give up gluten.</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>776</th>
      <td>positive</td>
      <td>I’m going to be stronger and things will be be...</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>777</th>
      <td>neutral</td>
      <td>Not too many because dad had just texted me sa...</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>778</th>
      <td>negative</td>
      <td>Why am I like this.</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>779</th>
      <td>negative</td>
      <td>My last final is in 3 hours and since I woke u...</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>780</th>
      <td>positive</td>
      <td>So much more to live.</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>781</th>
      <td>neutral</td>
      <td>The water was clear, lit by a single motion de...</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>782</th>
      <td>positive</td>
      <td>Practise problems are fine for me to focus on ...</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>783</th>
      <td>positive</td>
      <td>Sunday - more shops, found awesome outfit for ...</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>784</th>
      <td>neutral</td>
      <td>Wow.</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>785</th>
      <td>neutral</td>
      <td>Print out my fitness log on Sunday and sign it.</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>786</th>
      <td>neutral</td>
      <td>“Hey (me), what’s on your leg?” I keep saying ...</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>787</th>
      <td>negative</td>
      <td>Suddenly, I was reminded of another time I was...</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>788</th>
      <td>neutral</td>
      <td>She says I’m sleepwalking.</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>789</th>
      <td>negative</td>
      <td>He wanted to help but didn’t know how.</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>790</th>
      <td>negative</td>
      <td>Now it's time to get ready for bed because I h...</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>791</th>
      <td>neutral</td>
      <td>I grabbed my scratch paper from the quiz and h...</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>792</th>
      <td>neutral</td>
      <td>But I always have a habit of scraping their pl...</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>793</th>
      <td>neutral</td>
      <td>Reading girls feelings.</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>794</th>
      <td>positive</td>
      <td>The plate etching worked perfectly.</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>795</th>
      <td>neutral</td>
      <td>...They wrote that in the report.</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>796</th>
      <td>positive</td>
      <td>Her and I cleaned a lot and we watch Friends t...</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>797</th>
      <td>negative</td>
      <td>He had no idea I was angry with him for disapp...</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>798</th>
      <td>negative</td>
      <td>How did this happen?</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>799</th>
      <td>negative</td>
      <td>Next insecurity to write about is my penis size.</td>
      <td>valid</td>
    </tr>
    <tr>
      <th>800</th>
      <td>neutral</td>
      <td>Back on his lap.</td>
      <td>train</td>
    </tr>
    <tr>
      <th>801</th>
      <td>negative</td>
      <td>It’s mostly now I feel I’m just counting the d...</td>
      <td>train</td>
    </tr>
    <tr>
      <th>802</th>
      <td>neutral</td>
      <td>They tell me to grab the dogs call my boss and...</td>
      <td>train</td>
    </tr>
    <tr>
      <th>803</th>
      <td>negative</td>
      <td>What path is right for me, why should I have t...</td>
      <td>train</td>
    </tr>
  </tbody>
</table>
<p>804 rows × 3 columns</p>
</div>




```python
df.to_csv('data-split.csv')
```


```python

```
