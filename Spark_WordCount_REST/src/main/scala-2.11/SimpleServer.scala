import unfiltered.filter.Plan
import unfiltered.jetty.SocketPortBinding
import unfiltered.request._
import unfiltered.response._

/**
  * Created by sudhakar on 2/10/17.
  */

object SimplePlan extends Plan {
  def intent = {
    case req @ GET(Path("/get")) => {
      Ok ~> ResponseString(WordCount.count("Test #1: Test the Default word count program").mkString("\n"));
    }

    case req @ POST(Path("/get_custom")) => {
      val custom_string = Body.string(req)
      Ok ~> ResponseString(WordCount.count(custom_string).mkString("\n"))
    }
  }
}
object SimpleServer extends App {
  val bindingIP = SocketPortBinding(host = "localhost", port = 8080)
  unfiltered.jetty.Server.portBinding(bindingIP).plan(SimplePlan).run()
}
