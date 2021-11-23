# Numerical Linear Algebra

Resources for course on Numerical Linear Algebra. 
The resources include lecture notes, video links and numerical labs. 

## Jekyll Installation

https://www.youtube.com/watch?v=T1itpPvFWHI&list=PLLAZ4kZ9dFpOPV5C5Ay0pHaa0RJFhcmcB

or visit 
<br/>
https://jekyllrb.com

`clone` this repository to your Desktop

```bash
git clone https://github.com/NLAblog/numerical-linear-algebra.git 
```

Go to numerical linear algebra folder from your terminal/command line

```bash
cd Desktop/numerical-linear-algebra
```
and then run

```bash
bundle exec jekyll serve 
```

Then go to atom/VS-code and `open` numerical-linear-algebra

## Blog Posts 
Click `_posts` and create a file with the format "YYY-MM-DD-name-of-file.md to generate a markdown file.

Example: 2021-11-23-SVD.md

Once in the markdown file then copy paste the following to the top of the page

```markdown
---

layout: pos
title:  "Name of article"

---
```

**Syntax**

**Images**

`Save` your image in the images folder

then in the beginning of the page add
```markdown
img1 : /images/ur-image.png 
```

**Floating image**
```markdown
<div style="float: right;"> <img src="{{page.img1 | relative_url}}" height="220" width="400"></div>
```


**Centered Image**
```markdown
img2 : /images/ur-2nd-image.jpg
```
```markdown
<div style="float: right;"> <img src="{{page.img2 | relative_url}}" height="220" width="400"></div>
```

The top of your page should look like

```markdown
---

layout: pos
title:  "Name of article"
img1 : /images/ur-image.png 
img2 : /images/ur-2nd-image.jpg

---
```

## Code

{% highlight language %}

"your code"

{% endhighlight language %}

## References for other Syntax 
https://kramdown.gettalong.org/quickref.html#block-attributes

## Reference for Changing About/Details of home page
https://www.youtube.com/watch?v=ZtEbGztktvc&list=PLLAZ4kZ9dFpOPV5C5Ay0pHaa0RJFhcmcB&index=5

## Reference for changing default layout of Blog post
https://www.youtube.com/watch?v=bDQsGdCWv4I&list=PLLAZ4kZ9dFpOPV5C5Ay0pHaa0RJFhcmcB&index=12
