import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def snd_msj(remitente='jadg007p@gmail.com', 
            destinatario='jadg007p@gmail.com', 
            asunto='Termino ejecución Jupyter Hub',
            mensaje='Tarea terminada!'):
    # Configurar el servidor SMTP (ejemplo con Gmail)
    servidor_smtp = 'smtp.gmail.com'
    puerto = 587

    # Crear el mensaje
    correo = MIMEMultipart()
    correo['From'] = remitente
    correo['To'] = destinatario
    correo['Subject'] = asunto

    # Agregar el cuerpo del mensaje
    correo.attach(MIMEText(mensaje, 'plain'))

    try:
        # Establecer conexión con el servidor SMTP
        with smtplib.SMTP(servidor_smtp, puerto) as servidor:
            # Iniciar conexión segura
            servidor.starttls()

            # Iniciar sesión
            servidor.login(remitente, 'eirgasayhzpdiiko')

            # Enviar correo
            servidor.send_message(correo)

        print("Correo enviado exitosamente")

    except Exception as e:
        print(f"Error al enviar el correo: {e}")
