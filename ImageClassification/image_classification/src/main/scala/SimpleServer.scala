/**
  * Created by sudhakar on 2/12/17.
  */
import java.io.{File, ByteArrayInputStream}
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}
import javax.imageio.ImageIO
import java.util.Base64

import _root_.unfiltered.filter.Plan
import _root_.unfiltered.jetty.SocketPortBinding
import _root_.unfiltered.request.Body
import _root_.unfiltered.request.Method
import _root_.unfiltered.request.Path
import _root_.unfiltered.response.Ok
import _root_.unfiltered.response.ResponseString
import org.apache.commons.io.IOUtils
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
      Ok ~> ResponseString(IPApp.testImage("data/test/bibimap/1.jpg"))
    }

    case req @ POST(Path("/get_custom")) => {
      val imageByte = Base64.getDecoder.decode(Body.bytes(req))
      val bytes = new ByteArrayInputStream(imageByte)
      val image = ImageIO.read(bytes)
      ImageIO.write(image, "jpeg", new File("image.jpeg"))
      Ok ~> ResponseString(IPApp.testImage("image.jpeg"))
    }
  }
}
object SimpleServer extends App {
  val bindingIP = SocketPortBinding(host = "localhost", port = 8080)
  unfiltered.jetty.Server.portBinding(bindingIP).plan(SimplePlan).run()
}
