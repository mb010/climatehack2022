# climatehack2022
This is my attempt at [climatehack2022](https://climatehack.ai/).
The goal was to build a nowcasting model to predict the next 2 hours of satalite imagery (24 frames) given the previous hour (12 frames).

With my final attempts reaching 72nd overall I feel like this was a reasonable attempt with some clear avenues for improvement given enough time (which I sadly did not have).

Avenues for improvement:
- Use a larger (and potentially simpler) model.
- Improve training process by intentionally sampling multiple crops from a single source.
- Use a more apt loss: potentially a weighted combination of MSE and MSSIM.
- Train for longer: My training was cut short due to hardware related issues.
- Increase the batch size: This was memory limited with the model I chose to use, and the hardware I was training on.

The data engineering was by far the most challenging part of this project. I spent a significant portion of the evenings I was available for this project formatting the data into something more usable. If I had more time to spend on this project, or if the data engineering would have been more straight forward to implement, I believe I likely would have had time to test and implement these improvements.
This was a great experience, and I hope to be able to take part again next year with a more formidable attempt! 
Congratulations to the finalists, who all put in an immense amount of work!
