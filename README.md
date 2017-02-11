# ApacheSpark API
#### NOTE: Only for those who use Apache Spark in IDE by loading required libraries via a build tool.

Providing API capabilities using simple lightweight http server(Unfiltered) for Apache Spark applications. Instant testing (avoiding packaging and deploying using local machine's Spark & CLI) by using simple http wrapper for programatic execution in Intellij IDE.

##### The Problem: Provide api capabilities for spark programs which can accept http request and send response.

Http capabilities are achived using simple Http serving toolkit written for scala - [Unfiltered](http://unfiltered.ws/index.html).
Unfiltered provides:
* a server listening on a local port, wrap received http request and pass it to the application.
* a request mapping mechanism where you can define how http request should be handled.

From unfilter’s docs:
> * An _intent_ is a partial function for matching requests.
> * A _plan_ binds an intent to a particular server interface.

Demonstrating with Simple WorkCount program in Scala
##### GET
![](https://thumb.ibb.co/dxOiyv/GET.png)

##### POST
![](https://thumb.ibb.co/i7nXrF/POST.png)

### Todos

- Image Classification example

