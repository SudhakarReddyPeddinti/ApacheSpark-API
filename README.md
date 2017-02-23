# ApacheSpark - Scala - API
#### NOTE: Only for those who use Apache Spark in IDE by loading required libraries via a build tool.
Providing API capabilities using simple lightweight http server (like [flask](http://flask.pocoo.org)) for Apache Spark applications written in Scala. Instant testing (avoiding packaging and deploying using local machine's Spark & CLI) by using simple http wrapper for execution in Intellij IDE.

##### The Problem: Provide api capabilities for spark programs which can accept http request and send response.

API capabilities are achived using simple Http serving toolkit written for scala - [Unfiltered](http://unfiltered.ws/index.html).

Unfiltered provides:

* a server listening on a local port, wrap received http request and pass it to the application.
* a request mapping mechanism where you can define how http request should be handled.

From unfilter’s docs:
> * An _intent_ is a partial function for matching requests.
> * A _plan_ binds an intent to a particular server interface.

####Usage:
* Clone the git repository
* Import project in IntelliJ IDE
* Run SimpleServer
* open `localhost:8080` in any web browser

1. Demonstrating Simple WorkCount program in Postman client:

 * GET
![](http://i67.tinypic.com/358ozo4.png)

 * POST
![](http://i68.tinypic.com/34hhy8p.png)

2. Demonstrating Image Classification in Web Client:

 * Architecture
![Architecture](http://i68.tinypic.com/11alq9h.jpg)

 * Web client interface
![ImgClasfy](http://i65.tinypic.com/2drevsl.jpg)

### Todos

- [x] Word count example
- [x] Image Classification using spark example
- [ ] Image Classification comparison with tensorflow
